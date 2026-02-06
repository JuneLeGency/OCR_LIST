"""Chandra OCR backend - document OCR with layout-aware structured output."""

import time
from pathlib import Path
from typing import Union

from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("chandra-ocr")
class ChandraOCRBackend(OCRBackend):
    """
    Chandra OCR backend.

    Fine-tuned Qwen3-VL model (datalab-to/chandra) that produces
    layout-aware HTML output with bounding boxes, converted to markdown.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._manager = None

    @property
    def name(self) -> str:
        return "chandra-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.OFFLINE

    def load(self) -> None:
        if self._loaded:
            return

        from chandra.model import InferenceManager

        self._manager = InferenceManager(method="hf")
        self._loaded = True

    def unload(self) -> None:
        if self._manager is not None:
            # Free the underlying model
            if self._manager.model is not None:
                del self._manager.model
            del self._manager
            self._manager = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str = "请识别图片中的所有文字内容，保持原有格式。",
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            from chandra.model.schema import BatchInputItem

            pil_image = self._load_image(image)

            batch = [BatchInputItem(image=pil_image, prompt_type="ocr_layout")]
            results = self._manager.generate(batch)
            output = results[0]

            # Use markdown as the primary text output
            text = output.markdown or ""
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                text=text,
                model_name=self.name,
                inference_mode=self.inference_mode,
                tokens_generated=output.token_count,
                processing_time_ms=elapsed_ms,
                error="" if output.error else None,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                text="",
                model_name=self.name,
                inference_mode=self.inference_mode,
                processing_time_ms=elapsed_ms,
                error=str(e),
            )
