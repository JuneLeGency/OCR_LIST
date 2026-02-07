"""Allow running ``python -m ocr_engine`` to start the API server."""

import argparse

from .server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR Engine – OpenAI-compatible API server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--model", default=None, help="Model to preload at startup")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, model=args.model, log_level=args.log_level)


if __name__ == "__main__":
    main()
