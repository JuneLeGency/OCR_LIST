"""vLLM server client backend."""

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend


@register_backend("vllm")
class VLLMBackend(OCRBackend):
    """
    vLLM server client backend.

    Connects to a vLLM server via OpenAI-compatible API.
    Can be used with any model served by vLLM.

    Example vLLM server commands:
    - Qwen3-VL: vllm serve Qwen/Qwen3-VL-2B-Instruct
    - GLM-OCR:  vllm serve ZhipuAI/GLM-OCR
    - HunyuanOCR: vllm serve tencent/HunyuanOCR --trust-remote-code
    - DeepSeek-OCR: vllm serve deepseek-ai/DeepSeek-OCR-2 --trust-remote-code
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = None

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.VLLM

    def load(self) -> None:
        if self._loaded:
            return

        if not self.config.vllm_base_url:
            raise ValueError(
                "vLLM backend requires vllm_base_url in config. "
                "Example: vllm_base_url='http://localhost:8000/v1'"
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "vLLM backend requires openai package. "
                "Install with: pip install openai"
            )

        self.client = OpenAI(
            base_url=self.config.vllm_base_url,
            api_key=self.config.vllm_api_key,
        )
        self._loaded = True

    def unload(self) -> None:
        self.client = None
        self._loaded = False

    def _image_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        """Convert image to base64 data URL."""
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        b64_string = base64.b64encode(img_bytes).decode("utf-8")

        return f"data:image/png;base64,{b64_string}"

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            image_url = self._image_to_base64(image)
            model_name = self.config.get_vllm_model_name()

            # OpenAI-compatible chat completion with image
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

            text = response.choices[0].message.content or ""
            tokens_generated = response.usage.completion_tokens if response.usage else None

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return OCRResult(
                text=text,
                model_name=f"vllm:{model_name}",
                inference_mode=self.inference_mode,
                tokens_generated=tokens_generated,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                text="",
                model_name=f"vllm:{self.config.get_vllm_model_name()}",
                inference_mode=self.inference_mode,
                processing_time_ms=elapsed_ms,
                error=str(e),
            )

    def batch_ocr(
        self,
        images: list[Union[str, Path, Image.Image]],
        prompt: str,
    ) -> list[OCRResult]:
        """
        Batch OCR via vLLM.

        Note: vLLM server handles batching internally when receiving
        concurrent requests. For true batching, consider using
        async requests.
        """
        return [self.ocr(img, prompt) for img in images]
