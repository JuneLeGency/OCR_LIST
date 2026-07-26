"""Unlimited-OCR backend — proxies to the sidecar service."""

import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend

# Sidecar default; override with UNLIMITED_OCR_URL.
DEFAULT_SIDECAR_URL = "http://127.0.0.1:9085"


@register_backend("unlimited-ocr")
class UnlimitedOCRBackend(OCRBackend):
    """
    Unlimited-OCR (baidu/Unlimited-OCR, 3B) via HTTP sidecar.

    Unlike every other GPU backend here, this one holds **no** local weights.
    Unlimited-OCR's remote code is pinned to transformers 4.57.x and cannot run
    in this venv (transformers 5.x): the bundled modeling_deepseekv2.py fails on
    ``is_torch_fx_available``, and patching that just moves the failure to a
    device-side assert inside the vision encoder. So the model lives in
    ``~/Dev/unlimited-ocr`` with its own venv and is reached over the same
    OpenAI chat-completions contract this server exposes.

    Consequences for the caller:
    - VRAM (~7 GB) is charged to the sidecar process, not this one, so the
      ModelManager GPU-exclusion logic does not apply. It is listed in
      ``server._REMOTE_MODELS``.
    - The sidecar lazy-loads on first request (~4 s) and can be told to release
      VRAM via ``POST /unload``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._base_url = (
            config.vllm_base_url
            or os.environ.get("UNLIMITED_OCR_URL")
            or DEFAULT_SIDECAR_URL
        ).rstrip("/")

    @property
    def name(self) -> str:
        return "unlimited-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        # Remote HTTP inference — closest existing mode, keeps the enum stable.
        return InferenceMode.VLLM

    def load(self) -> None:
        """Probe the sidecar. Does not force the model to load over there."""
        if self._loaded:
            return
        import requests

        try:
            resp = requests.get(f"{self._base_url}/health", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Unlimited-OCR sidecar unreachable at {self._base_url} — {e}. "
                f"Start it with: ~/Dev/unlimited-ocr-server.sh start"
            )
        self._loaded = True

    def unload(self) -> None:
        """Ask the sidecar to release its VRAM; keep the process running."""
        if self._loaded:
            import requests

            try:
                requests.post(f"{self._base_url}/unload", timeout=30)
            except Exception:
                pass
        self._loaded = False

    def _image_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode()
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            import requests

            b64 = self._image_to_base64(image)
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": "unlimited-ocr",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
                timeout=self.config.extra_kwargs.get("timeout", 300),
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

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
