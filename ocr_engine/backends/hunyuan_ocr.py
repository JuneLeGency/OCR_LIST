"""HunyuanOCR backend."""

import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("hunyuan-ocr")
class HunyuanOCRBackend(OCRBackend):
    """
    HunyuanOCR backend.

    Based on Tencent-Hunyuan/HunyuanOCR.

    Requirements:
    - Uses transformers fork (hunyuan-vl-patch) for weight loading + generation_config fix
    - Two-step processing: apply_chat_template(tokenize=False) then processor()
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    @property
    def name(self) -> str:
        return "hunyuan-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.OFFLINE

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer, HunYuanVLForConditionalGeneration
            from transformers.models.hunyuan_vl import HunYuanVLProcessor, HunYuanVLImageProcessor
        except ImportError as e:
            raise ImportError(
                "HunyuanOCR requires HunYuanVLForConditionalGeneration.\n"
                "Ensure you're using the merged transformers from /tmp/transformers_merge\n"
                "Alternative: Use vLLM backend with: vllm serve tencent/HunyuanOCR --trust-remote-code\n"
                f"Original error: {e}"
            )

        model_path = self._get_model_path()

        # Manually construct processor to avoid video_processor loading issue.
        image_processor = HunYuanVLImageProcessor.from_pretrained(model_path)
        # The default max_pixels (4194304) produces too many visual tokens for
        # large images — the vision encoder's eager attention computes an
        # N²×heads attention matrix that OOMs (e.g. 16k tokens → 15.4 GiB).
        # Cap at 2M pixels (~7.7k patches, ~3.6 GiB attention) which fits
        # comfortably in a 32 GB GPU while preserving good OCR quality.
        image_processor.max_pixels = min(image_processor.max_pixels, 2_000_000)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = HunYuanVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
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
            img_path = self._get_image_path(image)

            # HunyuanOCR message format with system message
            messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Two-step processing per official example:
            # 1. Apply chat template WITHOUT tokenization
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 2. Process with images
            inputs = self.processor(
                text=[text],
                images=img,
                padding=True,
                return_tensors="pt",
            )

            device = next(self.model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    repetition_penalty=1.1,
                )

            # Trim input tokens from output
            input_ids = inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            tokens_generated = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0

            return OCRResult(
                text=text,
                model_name=self.name,
                inference_mode=self.inference_mode,
                tokens_generated=tokens_generated,
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
