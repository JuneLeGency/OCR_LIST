#!/usr/bin/env python3
"""Test all OCR backends."""

import sys
from pathlib import Path

from ocr_engine import OCREngine, list_all_models


def test_backend(model_name: str, image_path: str) -> bool:
    """Test a single backend."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)

    try:
        engine = OCREngine(model_name)
        print(f"Backend: {engine.backend_name}")
        print(f"Mode: {engine.inference_mode.value}")

        print("Loading model...")
        engine.load()
        print("Model loaded.")

        print("Running OCR...")
        result = engine.ocr(image_path, "请识别这张发票的金额。")

        print(f"Success: {result.success}")
        print(f"Time: {result.processing_time_ms:.0f}ms" if result.processing_time_ms else "Time: N/A")

        if result.error:
            print(f"Error: {result.error}")
            return False

        print(f"Text preview ({len(result.text)} chars):")
        print(result.text[:300])

        engine.unload()
        print("Model unloaded.")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    models = list_all_models()
    print("Available models:")
    print(f"  Offline: {models['offline']}")
    print(f"  vLLM: {models['vllm']}")

    # Test image
    test_image = "./inputs/25337000000272477597.jpg"
    if not Path(test_image).exists():
        print(f"\nError: Test image not found: {test_image}")
        return 1

    # Test offline models
    results = {}
    for model_name in ["qwen3-vl", "glm-ocr", "hunyuan-ocr"]:
        results[model_name] = test_backend(model_name, test_image)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    for model_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {model_name}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
