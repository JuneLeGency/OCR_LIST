"""RapidOCR backend - lightweight traditional OCR engine."""

import time
from pathlib import Path
from typing import Union

from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("rapidocr")
class RapidOCRBackend(OCRBackend):
    """
    RapidOCR backend.

    Lightweight traditional OCR engine based on PaddleOCR/ONNX.
    No GPU or large model required.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.engine = None

    @property
    def name(self) -> str:
        return "rapidocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.TRADITIONAL

    def load(self) -> None:
        if self._loaded:
            return

        from rapidocr import RapidOCR

        self.engine = RapidOCR()
        self._loaded = True

    def unload(self) -> None:
        if self.engine is not None:
            del self.engine
            self.engine = None
        self._loaded = False

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            image_path = self._get_image_path(image)
            result = self.engine(image_path)

            # RapidOCR returns a RapidOCROutput object with txts attribute
            text_parts = []
            if hasattr(result, "txts") and result.txts:
                text_parts = result.txts
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        text_parts.append(item[0])

            full_text = "\n".join(text_parts)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                text=full_text,
                model_name=self.name,
                inference_mode=self.inference_mode,
                processing_time_ms=elapsed_ms,
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
