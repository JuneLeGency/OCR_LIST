#!/usr/bin/env python3
"""
OCR Invoice Processing - Main Entry Point
Batch process invoices using the unified OCR engine.
"""

import sys
import argparse
from pathlib import Path

from ocr_engine import OCREngine


def process_with_model(model_name: str, input_dir: str, output_dir: str) -> None:
    """Process all invoices in input_dir with the given model."""
    input_path = Path(input_dir)
    image_files = sorted(
        input_path.glob("*.jpg")
    ) + sorted(
        input_path.glob("*.jpeg")
    ) + sorted(
        input_path.glob("*.png")
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"\nModel: {model_name}")
    print(f"Found {len(image_files)} images to process\n")

    engine = OCREngine(model_name)
    engine.load()

    for image_file in image_files:
        print(f"  Processing: {image_file.name} ... ", end="", flush=True)
        result = engine.ocr(str(image_file))
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            preview = result.text[:80].replace("\n", " ")
            time_str = f" ({result.processing_time_ms:.0f}ms)" if result.processing_time_ms else ""
            print(f"OK{time_str} - {preview}...")

        # Save result
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)
        result_file = out_path / f"{model_name}_{image_file.stem}.txt"
        result_file.write_text(result.text, encoding="utf-8")

    engine.unload()
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="OCR Invoice Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run RapidOCR (default)
  python main.py --rapidocr         # Explicitly use RapidOCR
  python main.py --model glm-ocr    # Use a specific model
  python main.py --all              # Run all offline models
  python main.py --evaluate         # Evaluate accuracy
  python main.py --list-models      # List available models
        """,
    )

    parser.add_argument(
        "--rapidocr",
        action="store_true",
        help="Use RapidOCR only (default, fastest)",
    )
    parser.add_argument(
        "--model",
        help="Use a specific model by name (e.g. glm-ocr, qwen3-vl)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all offline OCR models",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate accuracy against ground truth",
    )
    parser.add_argument(
        "--input-dir",
        default="./inputs",
        help="Directory containing invoice images (default: ./inputs)",
    )
    parser.add_argument(
        "--output-dir",
        default="./ocr_results",
        help="Directory for output results (default: ./ocr_results)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        models = OCREngine.list_models()
        print("Available models:")
        print("\nOffline:")
        for name in models["offline"]:
            print(f"  {name}")
        print("\nvLLM:")
        for name in models["vllm"]:
            print(f"  {name}")
        return 0

    # Default to RapidOCR if no mode specified
    if not (args.rapidocr or args.model or args.all or args.evaluate):
        args.rapidocr = True

    if args.model:
        print("=" * 80)
        print(f"STARTING {args.model.upper()} PROCESSING")
        print("=" * 80)
        try:
            process_with_model(args.model, args.input_dir, args.output_dir)
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.rapidocr:
        print("=" * 80)
        print("STARTING RAPIDOCR PROCESSING")
        print("=" * 80)
        try:
            process_with_model("rapidocr", args.input_dir, args.output_dir)
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.all:
        print("=" * 80)
        print("STARTING ALL OFFLINE MODELS PROCESSING")
        print("=" * 80)
        models = OCREngine.list_models()
        for model_name in models["offline"]:
            print("\n" + "-" * 80)
            try:
                process_with_model(model_name, args.input_dir, args.output_dir)
            except Exception as e:
                print(f"\nError with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if args.evaluate:
        print("\n" + "=" * 80)
        print("STARTING ACCURACY EVALUATION")
        print("=" * 80)
        try:
            from evaluate_ocr_accuracy import OCRAccuracyEvaluator

            results_dir = Path(args.output_dir)
            results_files = sorted(
                list(results_dir.glob("results_*.json"))
                + list(results_dir.glob("rapidocr_results_*.json")),
                reverse=True,
            )

            if not results_files:
                print("No results files found. Run processing first.")
                return 1

            latest_results = results_files[0]
            ground_truth_file = Path("./ground_truth.json")

            evaluator = OCRAccuracyEvaluator(
                str(latest_results),
                str(ground_truth_file) if ground_truth_file.exists() else None,
            )

            if evaluator.ground_truth:
                evaluator.evaluate_against_ground_truth()
            else:
                print(
                    "Ground truth file not found. Running consistency analysis instead.\n"
                )

            evaluator.evaluate_model_consistency()
            print("\nAccuracy evaluation completed successfully")
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
