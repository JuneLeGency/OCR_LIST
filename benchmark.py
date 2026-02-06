#!/usr/bin/env python3
"""
OCR Multi-Model Benchmark: Majority-Vote Consensus as Ground Truth.

Runs all OCR backends on invoice images, builds consensus via majority vote,
and evaluates each model's accuracy against the consensus.

Usage:
    python benchmark.py                          # Run all models + evaluate
    python benchmark.py --models rapidocr qwen3-vl
    python benchmark.py --skip-existing           # Skip models with existing results
    python benchmark.py --evaluate-only           # Only evaluate from existing results
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from ocr_engine.engine import OCREngine
from ocr_engine.config import SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# Phase 2: Amount extraction
# ---------------------------------------------------------------------------

def extract_amount(raw_text: str) -> Optional[float]:
    """Extract invoice amount from OCR raw text.

    Priority:
      1. ￥/¥ followed by number (standard invoice)
      2. 价税合计 ... ￥/¥ number (electronic invoice fallback)
      3. 个人现金支付：number (receipt fallback)
    """
    if not raw_text:
        return None

    patterns = [
        r"[（(小写）)]*[￥¥]\s*(\d+\.?\d*)",
        r"价税合计.*?[￥¥]\s*(\d+\.?\d*)",
        r"个人现金支付[：:]?\s*(\d+\.?\d*)",
    ]

    for pat in patterns:
        m = re.search(pat, raw_text)
        if m:
            try:
                return round(float(m.group(1)), 2)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Phase 1: Batch OCR
# ---------------------------------------------------------------------------

def collect_images(input_dir: Path) -> list[Path]:
    """Collect all image files from input directory, sorted by name."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    files: list[Path] = []
    for ext in exts:
        files.extend(input_dir.glob(ext))
    return sorted(files, key=lambda p: p.name)


def result_path(output_dir: Path, model: str) -> Path:
    return output_dir / f"benchmark_{model}.json"


def run_ocr_for_model(
    model_name: str,
    images: list[Path],
    output_dir: Path,
    skip_existing: bool = False,
) -> list[dict]:
    """Run OCR for a single model on all images and save results."""
    out_file = result_path(output_dir, model_name)

    if skip_existing and out_file.exists():
        print(f"  [skip] {model_name}: {out_file.name} already exists")
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  ({len(images)} images)")
    print(f"{'='*60}")

    engine = OCREngine(model_name)
    engine.load()

    results: list[dict] = []
    for idx, img in enumerate(images, 1):
        print(f"  [{idx}/{len(images)}] {img.name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            ocr_result = engine.ocr(str(img))
            if ocr_result.error:
                print(f"ERROR: {ocr_result.error}")
                results.append({
                    "filename": img.name,
                    "raw_text": "",
                    "extracted_amount": None,
                    "error": ocr_result.error,
                })
            else:
                amount = extract_amount(ocr_result.text)
                elapsed = time.time() - t0
                amt_str = f"¥{amount}" if amount is not None else "N/A"
                print(f"OK ({elapsed:.1f}s) amount={amt_str}")
                results.append({
                    "filename": img.name,
                    "raw_text": ocr_result.text,
                    "extracted_amount": amount,
                    "error": None,
                })
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append({
                "filename": img.name,
                "raw_text": "",
                "extracted_amount": None,
                "error": str(e),
            })

    engine.unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_file}")

    return results


# ---------------------------------------------------------------------------
# Phase 3: Majority-vote consensus
# ---------------------------------------------------------------------------

def build_consensus(
    all_results: dict[str, list[dict]],
) -> dict[str, dict]:
    """Build consensus from all model results.

    Returns:
        {filename: {consensus_amount, consensus_models, all_amounts}}
    """
    # Gather filenames across all models
    filenames: set[str] = set()
    for results in all_results.values():
        for r in results:
            filenames.add(r["filename"])

    consensus: dict[str, dict] = {}
    for fname in sorted(filenames):
        amounts: dict[str, Optional[float]] = {}
        for model, results in all_results.items():
            for r in results:
                if r["filename"] == fname:
                    amounts[model] = r["extracted_amount"]
                    break

        # Filter out None / error entries
        valid = {m: a for m, a in amounts.items() if a is not None}

        if not valid:
            consensus[fname] = {
                "consensus_amount": None,
                "consensus_models": [],
                "all_amounts": amounts,
                "status": "no_data",
            }
            continue

        # Group by rounded value
        counter: Counter = Counter()
        for a in valid.values():
            counter[round(a, 2)] += 1

        most_common_val, most_common_count = counter.most_common(1)[0]

        if most_common_count < 2:
            # No two models agree
            consensus[fname] = {
                "consensus_amount": None,
                "consensus_models": [],
                "all_amounts": amounts,
                "status": "disputed",
            }
        else:
            supporting = [
                m for m, a in valid.items()
                if round(a, 2) == most_common_val
            ]
            consensus[fname] = {
                "consensus_amount": most_common_val,
                "consensus_models": supporting,
                "all_amounts": amounts,
                "status": "consensus",
            }

    return consensus


# ---------------------------------------------------------------------------
# Phase 4: Evaluation report
# ---------------------------------------------------------------------------

def evaluate(
    all_results: dict[str, list[dict]],
    consensus: dict[str, dict],
) -> dict:
    """Evaluate each model against consensus and print report."""
    models = sorted(all_results.keys())

    # Per-model stats
    stats: dict[str, dict] = {}
    for model in models:
        match = mismatch = error = skip = 0
        for r in all_results[model]:
            fname = r["filename"]
            c = consensus.get(fname)
            if c is None or c["consensus_amount"] is None:
                skip += 1
                continue
            if r["extracted_amount"] is None or r.get("error"):
                error += 1
                continue
            if abs(r["extracted_amount"] - c["consensus_amount"]) <= 0.01:
                match += 1
            else:
                mismatch += 1
        denom = match + mismatch
        accuracy = (match / denom * 100) if denom > 0 else 0.0
        stats[model] = {
            "match": match,
            "mismatch": mismatch,
            "error": error,
            "skipped_disputed": skip,
            "accuracy": round(accuracy, 2),
        }

    # ---- Terminal table ----
    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS (consensus-based)")
    print("=" * 72)
    header = f"{'Model':<18} {'Accuracy':>8} {'Match':>6} {'Mismatch':>9} {'Error':>6} {'Skip':>6}"
    print(header)
    print("-" * 72)
    for model in models:
        s = stats[model]
        print(
            f"{model:<18} {s['accuracy']:>7.1f}% {s['match']:>6} "
            f"{s['mismatch']:>9} {s['error']:>6} {s['skipped_disputed']:>6}"
        )
    print("-" * 72)

    # ---- Per-image detail (disagreements) ----
    disputed_files = [
        fname for fname, c in consensus.items()
        if c["status"] == "disputed"
    ]
    mismatch_files = []
    for fname, c in consensus.items():
        if c["consensus_amount"] is None:
            continue
        for model in models:
            for r in all_results[model]:
                if r["filename"] == fname and r["extracted_amount"] is not None:
                    if abs(r["extracted_amount"] - c["consensus_amount"]) > 0.01:
                        mismatch_files.append(fname)
                        break
            else:
                continue
            break

    if disputed_files or mismatch_files:
        print(f"\n  Disputed images (no consensus): {len(disputed_files)}")
        for fname in disputed_files:
            c = consensus[fname]
            amounts_str = ", ".join(
                f"{m}={a}" for m, a in c["all_amounts"].items()
            )
            print(f"    * {fname}: {amounts_str}")

        print(f"\n  Images with model disagreements: {len(mismatch_files)}")
        for fname in sorted(set(mismatch_files)):
            c = consensus[fname]
            amounts_str = ", ".join(
                f"{m}={a}" for m, a in c["all_amounts"].items()
            )
            print(f"    ! {fname}: consensus={c['consensus_amount']}  |  {amounts_str}")

    # ---- Consensus summary ----
    n_total = len(consensus)
    n_consensus = sum(1 for c in consensus.values() if c["status"] == "consensus")
    n_disputed = sum(1 for c in consensus.values() if c["status"] == "disputed")
    n_nodata = sum(1 for c in consensus.values() if c["status"] == "no_data")
    print(f"\n  Consensus: {n_consensus}/{n_total} images")
    print(f"  Disputed:  {n_disputed}/{n_total}")
    print(f"  No data:   {n_nodata}/{n_total}")

    return {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "stats": stats,
        "consensus": {
            fname: {
                "consensus_amount": c["consensus_amount"],
                "consensus_models": c["consensus_models"],
                "status": c["status"],
                "all_amounts": {
                    m: a for m, a in c["all_amounts"].items()
                },
            }
            for fname, c in consensus.items()
        },
        "summary": {
            "total_images": n_total,
            "consensus_count": n_consensus,
            "disputed_count": n_disputed,
            "no_data_count": n_nodata,
        },
    }


def update_ground_truth(consensus: dict[str, dict], gt_path: Path) -> None:
    """Update ground_truth.json with consensus amounts."""
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
    else:
        gt = {}

    updated = 0
    for fname, c in consensus.items():
        if c["consensus_amount"] is not None:
            gt[fname] = c["consensus_amount"]
            updated += 1

    # Remove instruction keys
    gt.pop("_instructions", None)
    gt.pop("_example", None)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print(f"\n  Updated {gt_path} with {updated} consensus values")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR multi-model benchmark with majority-vote consensus",
    )
    parser.add_argument(
        "--models", nargs="+",
        help=f"Models to run (default: all). Choices: {list(SUPPORTED_MODELS.keys())}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models whose benchmark_<model>.json already exists",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip OCR, evaluate from existing benchmark_*.json files",
    )
    parser.add_argument(
        "--input-dir", default="./inputs",
        help="Directory containing invoice images (default: ./inputs)",
    )
    parser.add_argument(
        "--output-dir", default="./ocr_results",
        help="Directory for output results (default: ./ocr_results)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models or list(SUPPORTED_MODELS.keys())

    # Validate model names
    for m in model_names:
        if m not in SUPPORTED_MODELS:
            print(f"Error: unknown model '{m}'. Available: {list(SUPPORTED_MODELS.keys())}")
            return 1

    # Phase 1: Run OCR (unless --evaluate-only)
    all_results: dict[str, list[dict]] = {}

    if not args.evaluate_only:
        images = collect_images(input_dir)
        if not images:
            print(f"No images found in {input_dir}")
            return 1
        print(f"Found {len(images)} images in {input_dir}")

        for model_name in model_names:
            results = run_ocr_for_model(
                model_name, images, output_dir,
                skip_existing=args.skip_existing,
            )
            all_results[model_name] = results
    else:
        # Load existing results
        for model_name in model_names:
            rp = result_path(output_dir, model_name)
            if rp.exists():
                with open(rp, encoding="utf-8") as f:
                    all_results[model_name] = json.load(f)
                print(f"  Loaded {rp.name} ({len(all_results[model_name])} entries)")
            else:
                print(f"  [warn] No results file for {model_name}: {rp.name}")

    if not all_results:
        print("No results to evaluate.")
        return 1

    # Phase 3: Build consensus
    consensus = build_consensus(all_results)

    # Phase 4: Evaluate and report
    report = evaluate(all_results, consensus)

    # Save JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"benchmark_report_{ts}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved → {report_file}")

    # Update ground_truth.json
    gt_path = Path("ground_truth.json")
    update_ground_truth(consensus, gt_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
