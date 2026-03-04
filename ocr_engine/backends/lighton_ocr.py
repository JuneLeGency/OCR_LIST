"""LightOnOCR-2 backend."""

import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("lighton-ocr")
class LightOnOCRBackend(OCRBackend):
    """
    LightOnOCR-2 backend.

    Based on lightonai/LightOnOCR-2-1B.
    Pixtral ViT encoder + Qwen3 decoder (~1B params).
    No Conv3D workaround needed (uses Pixtral ViT, not Qwen3-VL Conv3D).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    @property
    def name(self) -> str:
        return "lighton-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.OFFLINE

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import (
            LightOnOcrForConditionalGeneration,
            LightOnOcrProcessor,
        )

        model_id = self.config.model_id
        dtype = self._get_torch_dtype()

        self.processor = LightOnOcrProcessor.from_pretrained(model_id)
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )

        self.model.eval()
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

            content = [{"type": "image", "image": img}]
            if prompt:
                content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(device=self.model.device, dtype=self.model.dtype)
                if v.is_floating_point()
                else v.to(self.model.device)
                for k, v in inputs.items()
            }

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
