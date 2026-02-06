"""Tesseract OCR backend - traditional OCR via Google Tesseract."""

import time
from pathlib import Path
from typing import Union

from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("tesseract")
class TesseractOCRBackend(OCRBackend):
    """
    Tesseract OCR backend.

    Requires system package tesseract-ocr and language packs
    (e.g. tesseract-ocr-chi-sim for Simplified Chinese).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.TRADITIONAL

    def load(self) -> None:
        if self._loaded:
            return

        import pytesseract
        # Verify tesseract binary is accessible
        pytesseract.get_tesseract_version()
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str = "",
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            import pytesseract

            pil_image = self._load_image(image)

            # Use Chinese Simplified + English by default
            text = pytesseract.image_to_string(pil_image, lang="chi_sim+eng")
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                text=text.strip(),
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
