"""OCR model backends."""

from typing import Type

from ..base import ModelConfig, OCRBackend, InferenceMode

# Backend registry
_BACKENDS: dict[str, Type[OCRBackend]] = {}


def register_backend(name: str):
    """Decorator to register a backend class."""
    def decorator(cls: Type[OCRBackend]):
        _BACKENDS[name] = cls
        return cls
    return decorator


def get_backend_class(name: str) -> Type[OCRBackend]:
    """Get backend class by name."""
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. "
            f"Available: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name]


def create_backend(name: str, config: ModelConfig) -> OCRBackend:
    """Create a backend instance."""
    cls = get_backend_class(name)
    return cls(config)


def list_backends() -> list[str]:
    """List available backend names."""
    return list(_BACKENDS.keys())


# Import backends to register them
from .qwen3_vl import Qwen3VLBackend
from .glm_ocr import GLMOCRBackend
from .hunyuan_ocr import HunyuanOCRBackend
from .deepseek_ocr import DeepSeekOCRBackend
from .vllm_client import VLLMBackend
from .rapidocr import RapidOCRBackend

__all__ = [
    "register_backend",
    "get_backend_class",
    "create_backend",
    "list_backends",
    "Qwen3VLBackend",
    "GLMOCRBackend",
    "HunyuanOCRBackend",
    "DeepSeekOCRBackend",
    "VLLMBackend",
    "RapidOCRBackend",
]
