"""
Unified OCR Engine for VLM models.

Supports both offline inference and vLLM server mode.

Models:
- Qwen3-VL (Qwen/Qwen3-VL-2B-Instruct)
- GLM-OCR (ZhipuAI/GLM-OCR)
- HunyuanOCR (Tencent-Hunyuan/HunyuanOCR)
- DeepSeek-OCR-2 (deepseek-ai/DeepSeek-OCR-2)
- RapidOCR (lightweight traditional OCR, no GPU required)

Example:
    from ocr_engine import OCREngine, ocr

    # Quick usage
    text = ocr("invoice.jpg", model="qwen3-vl")

    # With engine instance
    engine = OCREngine("glm-ocr")
    result = engine.ocr("invoice.jpg")
    print(result.text)

    # Context manager (auto cleanup)
    with OCREngine("hunyuan-ocr") as engine:
        result = engine.ocr("invoice.jpg")

    # vLLM server mode
    engine = OCREngine("glm-ocr-vllm")
    result = engine.ocr("invoice.jpg")
"""

from .base import ModelConfig, OCRResult, InferenceMode, OCRBackend
from .config import (
    get_model_config,
    list_all_models,
    create_vllm_config,
    SUPPORTED_MODELS,
    VLLM_CONFIGS,
)
from .engine import OCREngine, ocr, batch_ocr

__version__ = "0.2.0"

__all__ = [
    # Main classes
    "OCREngine",
    "ModelConfig",
    "OCRResult",
    "InferenceMode",
    "OCRBackend",
    # Convenience functions
    "ocr",
    "batch_ocr",
    # Configuration
    "get_model_config",
    "list_all_models",
    "create_vllm_config",
    "SUPPORTED_MODELS",
    "VLLM_CONFIGS",
]
