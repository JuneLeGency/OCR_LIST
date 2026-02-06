"""Base classes and interfaces for OCR backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from PIL import Image


class InferenceMode(Enum):
    """Inference mode for OCR models."""
    OFFLINE = "offline"        # Direct VLM model loading
    VLLM = "vllm"              # vLLM server API
    TRADITIONAL = "traditional"  # Traditional OCR (RapidOCR etc.)


@dataclass
class ModelConfig:
    """Configuration for a VLM OCR model."""

    name: str
    model_id: str

    # Model loading settings
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: Optional[str] = None  # None = auto, "eager", "flash_attention_2"

    # Generation settings
    max_new_tokens: int = 8192
    do_sample: bool = False
    temperature: float = 0.0

    # Model-specific settings
    extra_kwargs: dict = field(default_factory=dict)

    # vLLM server settings
    vllm_base_url: Optional[str] = None
    vllm_api_key: str = "EMPTY"
    vllm_model_name: Optional[str] = None  # Override model name for vLLM

    # Cache settings
    cache_dir: Optional[str] = None

    def get_vllm_model_name(self) -> str:
        """Get model name for vLLM API."""
        return self.vllm_model_name or self.model_id


@dataclass
class OCRResult:
    """Result of OCR operation."""

    text: str
    model_name: str
    inference_mode: InferenceMode

    # Optional metadata
    tokens_generated: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class OCRBackend(ABC):
    """Abstract base class for OCR model backends."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    @property
    @abstractmethod
    def inference_mode(self) -> InferenceMode:
        """Inference mode (offline or vllm)."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model and processor. Should be idempotent."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model to free memory."""
        pass

    @abstractmethod
    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
    ) -> OCRResult:
        """
        Perform OCR on a single image.

        Args:
            image: Image path or PIL Image
            prompt: OCR prompt/instruction

        Returns:
            OCRResult with extracted text
        """
        pass

    def batch_ocr(
        self,
        images: list[Union[str, Path, Image.Image]],
        prompt: str,
    ) -> list[OCRResult]:
        """
        Perform OCR on multiple images.
        Default implementation processes sequentially.
        Subclasses may override for batch optimization.
        """
        return [self.ocr(img, prompt) for img in images]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            self.load()

    # Helper methods for subclasses

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def _get_image_path(self, image: Union[str, Path, Image.Image]) -> str:
        """Get image path, saving PIL Image to temp file if needed."""
        if isinstance(image, (str, Path)):
            return str(image)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            return f.name

    def _get_torch_dtype(self):
        """Get torch dtype from config."""
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        return dtype_map.get(self.config.torch_dtype, torch.bfloat16)

    def _get_cache_dir(self) -> str:
        """Get model cache directory."""
        import os
        return self.config.cache_dir or os.path.expanduser("~/.cache/modelscope/hub/models")

    def _get_model_path(self) -> str:
        """Get local model path from cache."""
        import os
        return os.path.join(self._get_cache_dir(), self.config.model_id)
