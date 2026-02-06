"""DeepSeek-OCR backend."""

import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("deepseek-ocr")
class DeepSeekOCRBackend(OCRBackend):
    """
    DeepSeek-OCR-2 backend.

    Based on deepseek-ai/DeepSeek-OCR-2.

    Requirements:
    - Requires trust_remote_code=True
    - Uses custom infer() method instead of generate()
    - Prompt must start with <image> tag
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    @property
    def name(self) -> str:
        return "deepseek-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.OFFLINE

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import AutoModel, AutoTokenizer

        model_path = self._get_model_path()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=True,
            **self.config.extra_kwargs,
        )
        # DeepSeek requires explicit CUDA placement
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        self._loaded = True

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            image_path = self._get_image_path(image)

            # DeepSeek requires <image> tag prefix
            if not prompt.startswith("<image>"):
                prompt = f"<image>\n{prompt}"

            # DeepSeek uses its own infer() method
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=tmp_dir,
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=False,
                    eval_mode=True,
                )

            # Result may be a list
            if isinstance(result, list):
                text = "\n".join(str(r) for r in result)
            else:
                text = str(result)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                text=text,
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
