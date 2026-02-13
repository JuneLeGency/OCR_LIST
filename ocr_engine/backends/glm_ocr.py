"""GLM-OCR backend."""

import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("glm-ocr")
class GLMOCRBackend(OCRBackend):
    """
    GLM-OCR backend.

    Based on ZhipuAI/GLM-OCR.
    Works with transformers main branch (5.x).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    @property
    def name(self) -> str:
        return "glm-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.OFFLINE

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = self._get_model_path()
        dtype = self._get_torch_dtype()

        self.processor = AutoProcessor.from_pretrained(model_path)
        # The default longest_edge (9633792 ≈ 9.6M pixels) means large images
        # pass through at full resolution, producing ~24k visual tokens for a
        # 3000×2000 image vs ~4k for a 1080×740 image (6× slower).
        # Cap at 4M pixels to keep large-image patches around ~10k, reducing
        # inference from ~64s to ~25s while preserving OCR quality.
        ip = self.processor.image_processor
        ip.size = {
            "shortest_edge": ip.size["shortest_edge"],
            "longest_edge": min(ip.size["longest_edge"], 4_000_000),
        }
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()

        from ..compat import patch_conv3d_for_blackwell
        patch_conv3d_for_blackwell(self.model)

        self._loaded = True

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
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
            img = self._load_image(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            # GLM-OCR specific: remove token_type_ids
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0][input_len:]
            text = self.processor.decode(output_ids, skip_special_tokens=True)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                text=text,
                model_name=self.name,
                inference_mode=self.inference_mode,
                tokens_generated=len(output_ids),
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
