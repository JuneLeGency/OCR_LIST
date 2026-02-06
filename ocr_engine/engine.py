"""Unified OCR Engine supporting multiple VLM models."""

from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from .config import (
    get_model_config,
    get_backend_name,
    list_all_models,
    SUPPORTED_MODELS,
    VLLM_CONFIGS,
)
from .backends import create_backend, list_backends


class OCREngine:
    """
    Unified OCR engine supporting multiple VLM models.

    Supports both offline inference (direct model loading) and
    vLLM server mode (OpenAI-compatible API).

    Example usage:

        # Offline inference
        engine = OCREngine("qwen3-vl")
        result = engine.ocr("invoice.jpg", "Extract all text")
        print(result.text)

        # vLLM server mode
        engine = OCREngine("glm-ocr-vllm")
        result = engine.ocr("invoice.jpg", "Extract all text")
        print(result.text)

        # Custom vLLM server
        from ocr_engine.config import create_vllm_config
        config = create_vllm_config(
            model_id="ZhipuAI/GLM-OCR",
            base_url="http://my-server:8000/v1",
        )
        engine = OCREngine(config=config)
        result = engine.ocr("invoice.jpg")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize OCR engine.

        Args:
            model_name: Model identifier. Options:
                Offline: 'qwen3-vl', 'glm-ocr', 'hunyuan-ocr', 'deepseek-ocr'
                vLLM: 'qwen3-vl-vllm', 'glm-ocr-vllm', 'hunyuan-ocr-vllm', 'deepseek-ocr-vllm'
            config: Optional custom ModelConfig (overrides model_name)
            cache_dir: Model cache directory (for offline models)

        Either model_name or config must be provided.
        """
        if config is None and model_name is None:
            model_name = "qwen3-vl"  # Default model

        if config is not None:
            self.config = config
            # Determine backend from config
            if config.vllm_base_url:
                self._backend_name = "vllm"
            else:
                # Try to match config to a known model
                for name, cfg in SUPPORTED_MODELS.items():
                    if cfg.model_id == config.model_id:
                        self._backend_name = name
                        break
                else:
                    raise ValueError(
                        f"Cannot determine backend for model_id: {config.model_id}. "
                        "Please specify model_name or use a known model_id."
                    )
        else:
            self.config = get_model_config(model_name)
            self._backend_name = get_backend_name(model_name)

        # Apply cache_dir if provided
        if cache_dir:
            self.config.cache_dir = cache_dir

        self._backend: Optional[OCRBackend] = None
        self._model_name = model_name or self.config.name

    @property
    def model_name(self) -> str:
        """Model name identifier."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Backend name (e.g., 'qwen3-vl', 'vllm')."""
        return self._backend_name

    @property
    def inference_mode(self) -> InferenceMode:
        """Current inference mode."""
        if self._backend:
            return self._backend.inference_mode
        return InferenceMode.VLLM if self.config.vllm_base_url else InferenceMode.OFFLINE

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._backend is not None and self._backend.is_loaded

    def load(self) -> "OCREngine":
        """Load model. Returns self for chaining."""
        if self._backend is None:
            self._backend = create_backend(self._backend_name, self.config)
        self._backend.load()
        return self

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str = "请识别图片中的所有文字内容，保持原有格式。",
    ) -> OCRResult:
        """
        Perform OCR on an image.

        Args:
            image: Image path or PIL Image
            prompt: OCR prompt/instruction

        Returns:
            OCRResult with extracted text and metadata
        """
        if self._backend is None:
            self.load()
        return self._backend.ocr(image, prompt)

    def batch_ocr(
        self,
        images: list[Union[str, Path, Image.Image]],
        prompt: str = "请识别图片中的所有文字内容，保持原有格式。",
    ) -> list[OCRResult]:
        """
        Perform OCR on multiple images.

        Args:
            images: List of image paths or PIL Images
            prompt: OCR prompt/instruction

        Returns:
            List of OCRResult objects
        """
        if self._backend is None:
            self.load()
        return self._backend.batch_ocr(images, prompt)

    def __enter__(self) -> "OCREngine":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.unload()

    @classmethod
    def list_models(cls) -> dict[str, list[str]]:
        """List available models grouped by type."""
        return list_all_models()

    @classmethod
    def list_backends(cls) -> list[str]:
        """List available backend names."""
        return list_backends()


# Convenience functions for quick usage
def ocr(
    image: Union[str, Path, Image.Image],
    prompt: str = "请识别图片中的所有文字内容，保持原有格式。",
    model: str = "qwen3-vl",
) -> str:
    """
    Quick OCR function.

    Args:
        image: Image path or PIL Image
        prompt: OCR prompt
        model: Model name

    Returns:
        Extracted text
    """
    with OCREngine(model) as engine:
        result = engine.ocr(image, prompt)
        if result.error:
            raise RuntimeError(f"OCR failed: {result.error}")
        return result.text


def batch_ocr(
    images: list[Union[str, Path, Image.Image]],
    prompt: str = "请识别图片中的所有文字内容，保持原有格式。",
    model: str = "qwen3-vl",
) -> list[str]:
    """
    Quick batch OCR function.

    Args:
        images: List of image paths or PIL Images
        prompt: OCR prompt
        model: Model name

    Returns:
        List of extracted texts
    """
    with OCREngine(model) as engine:
        results = engine.batch_ocr(images, prompt)
        texts = []
        for result in results:
            if result.error:
                texts.append(f"ERROR: {result.error}")
            else:
                texts.append(result.text)
        return texts
