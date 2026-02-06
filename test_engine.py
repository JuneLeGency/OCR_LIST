#!/usr/bin/env python3
"""Test script for unified OCR engine."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_engine import OCREngine, SUPPORTED_MODELS


def test_list_models():
    """Test listing available models."""
    print("Available models:")
    for name, config in SUPPORTED_MODELS.items():
        print(f"  {name}: {config.model_id}")
    print()


def test_single_model(model_name: str, image_path: str):
    """Test a single model on an image."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    try:
        with OCREngine(model_name=model_name) as engine:
            print(f"Loading {engine.config.model_id}...")

            print(f"Running OCR on {image_path}...")
            result = engine.ocr(image_path)

            if result.error:
                print(f"OCR Error: {result.error}")
                return False

            text = result.text
            print(f"\nResult ({len(text)} chars, {result.processing_time_ms:.0f}ms):")
            print("-" * 40)
            print(text[:2000] + "..." if len(text) > 2000 else text)
            print("-" * 40)
            return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Image to test")
    parser.add_argument("-m", "--model", default="qwen3-vl", help="Model to test")
    parser.add_argument("--all", action="store_true", help="Test all models")
    args = parser.parse_args()

    test_list_models()

    if not args.image:
        # Find a test image
        inputs_dir = Path(__file__).parent / "inputs"
        if inputs_dir.exists():
            images = list(inputs_dir.glob("*.jpg")) + list(inputs_dir.glob("*.png"))
            if images:
                args.image = str(images[0])
                print(f"Using test image: {args.image}")

    if not args.image:
        print("No image specified and no test images found in inputs/")
        return 1

    if args.all:
        results = {}
        for model_name in SUPPORTED_MODELS:
            results[model_name] = test_single_model(model_name, args.image)

        print("\n" + "=" * 60)
        print("Summary:")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
    else:
        success = test_single_model(args.model, args.image)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
