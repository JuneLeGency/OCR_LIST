"""Command-line interface for OCR engine."""

import argparse
import json
import sys
from pathlib import Path

from .engine import OCREngine
from .config import SUPPORTED_MODELS, VLLM_CONFIGS, list_all_models


def _run_ocr(args) -> int:
    """Execute the OCR (run) subcommand."""
    if args.list_models:
        models = list_all_models()
        print("Available models:")
        print("\nOffline (direct model loading):")
        for name in models["offline"]:
            config = SUPPORTED_MODELS[name]
            print(f"  {name}: {config.model_id}")
        print("\nvLLM (server mode):")
        for name in models["vllm"]:
            config = VLLM_CONFIGS[name]
            print(f"  {name}: {config.model_id} @ {config.vllm_base_url}")
        return 0

    if not args.images:
        print("Error: No images specified. Use --list-models to see available models.",
              file=sys.stderr)
        return 1

    # Initialize engine
    try:
        engine = OCREngine(model_name=args.model)
        engine.load()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Model: {engine.model_name}", file=sys.stderr)
        print(f"Backend: {engine.backend_name}", file=sys.stderr)
        print(f"Mode: {engine.inference_mode.value}", file=sys.stderr)

    results = []
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(f"Warning: {image_path} not found, skipping", file=sys.stderr)
            continue

        if args.verbose:
            print(f"Processing: {path}", file=sys.stderr)

        result = engine.ocr(path, prompt=args.prompt)

        result_dict = {
            "file": str(path),
            "text": result.text,
            "model": result.model_name,
            "mode": result.inference_mode.value,
        }

        if result.error:
            result_dict["error"] = result.error

        if args.verbose and result.processing_time_ms:
            result_dict["time_ms"] = round(result.processing_time_ms, 2)
            print(f"  Time: {result.processing_time_ms:.0f}ms", file=sys.stderr)

        if result.tokens_generated:
            result_dict["tokens"] = result.tokens_generated

        results.append(result_dict)

    # Output
    if args.json:
        output = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        output_lines = []
        for r in results:
            if len(results) > 1:
                output_lines.append(f"=== {r['file']} ===")
            if r.get("error"):
                output_lines.append(f"ERROR: {r['error']}")
            else:
                output_lines.append(r["text"])
            if len(results) > 1:
                output_lines.append("")
        output = "\n".join(output_lines)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Cleanup
    engine.unload()
    return 0


def _run_serve(args) -> int:
    """Execute the serve subcommand."""
    from .server import run_server

    run_server(
        host=args.host,
        port=args.port,
        model=args.model,
        log_level=args.log_level,
    )
    return 0


def _run_web(args) -> int:
    """Execute the web subcommand."""
    from .web import run_web

    run_web(host=args.host, port=args.port)
    return 0


def _add_ocr_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OCR-specific arguments to a parser."""
    all_models = list(SUPPORTED_MODELS.keys()) + list(VLLM_CONFIGS.keys())

    parser.add_argument(
        "images",
        nargs="*",
        help="Image files to process",
    )
    parser.add_argument(
        "-m", "--model",
        choices=all_models,
        default="qwen3-vl",
        help="OCR model to use (default: qwen3-vl)",
    )
    parser.add_argument(
        "-p", "--prompt",
        default="请识别图片中的所有文字内容，保持原有格式。",
        help="OCR prompt",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with timing info",
    )


def main():
    # Detect whether the first positional arg is a known subcommand.
    # If not, fall back to the legacy OCR-only interface so that
    # ``ocr image.png`` keeps working without requiring ``ocr run``.
    subcommands = {"serve", "run", "web"}
    use_subcommand = len(sys.argv) > 1 and sys.argv[1] in subcommands

    if not use_subcommand:
        # Legacy / shorthand mode: ``ocr [options] images...``
        parser = argparse.ArgumentParser(
            description="Unified OCR Engine for VLM models",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  ocr image.png                        # OCR with default model (qwen3-vl)
  ocr -m glm-ocr image.png             # OCR with GLM-OCR
  ocr -m hunyuan-ocr invoice.jpg       # OCR with HunyuanOCR
  ocr -m deepseek-ocr *.png            # Batch OCR with DeepSeek
  ocr -m glm-ocr-vllm image.png        # OCR via vLLM server
  ocr --list-models                    # List available models
  ocr -p "提取表格内容" invoice.jpg     # Custom prompt
  ocr -o output.json --json *.jpg      # JSON output to file
  ocr serve                            # Start API server
  ocr serve --port 9090 --model rapidocr
            """,
        )
        _add_ocr_arguments(parser)
        args = parser.parse_args()
        return _run_ocr(args)

    # Subcommand mode
    parser = argparse.ArgumentParser(
        description="Unified OCR Engine for VLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- serve ---
    serve_parser = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--model", default=None, help="Model to preload at startup")
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )

    # --- run ---
    run_parser = sub.add_parser("run", help="Run OCR on images (explicit subcommand)")
    _add_ocr_arguments(run_parser)

    # --- web ---
    web_parser = sub.add_parser("web", help="Start Gradio web UI for model comparison")
    web_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    web_parser.add_argument("--port", type=int, default=9090, help="Bind port (default: 9090)")

    args = parser.parse_args()

    if args.command == "serve":
        return _run_serve(args)
    elif args.command == "run":
        return _run_ocr(args)
    elif args.command == "web":
        return _run_web(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
