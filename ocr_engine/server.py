"""OpenAI-compatible API server for OCR engine.

Exposes all OCR backends (including RapidOCR) via a standard
``/v1/chat/completions`` endpoint.  GPU models are exclusive –
only one may be loaded at a time – while RapidOCR (CPU-only)
can always coexist.

Usage:
    python -m ocr_engine --port 8000 --model rapidocr
    ocr serve --port 8000 --model rapidocr
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from .config import SUPPORTED_MODELS
from .engine import OCREngine

# Estimated VRAM usage per GPU model (MB).  Used as a safety budget check
# before loading.  Values are conservative upper bounds measured empirically.
_MODEL_VRAM_MB: dict[str, int] = {
    "glm-ocr":      2600,
    "hunyuan-ocr":  2200,
    "lighton-ocr":  2200,
    "qwen3-vl":     4500,
    "firered-ocr":  4500,
    "deepseek-ocr": 7000,
    "dots-ocr":     6500,
    "chandra-ocr": 18000,
    # Charged to the sidecar process, listed for reference only — see
    # _REMOTE_MODELS, which exempts it from the local budget check.
    "unlimited-ocr": 7000,
}
# Minimum free VRAM (MB) to keep after loading, for inference workspace
_VRAM_RESERVE_MB = 300

logger = logging.getLogger("ocr_engine.server")

# ---------------------------------------------------------------------------
# Models that only use CPU (can coexist with any GPU model)
# ---------------------------------------------------------------------------
_CPU_ONLY_MODELS = {"rapidocr"}

# ---------------------------------------------------------------------------
# Models served by a sidecar process over HTTP.  Their weights never enter this
# process, so the single-GPU-model exclusion and the local VRAM budget check
# must not apply to them — otherwise loading one would needlessly evict the
# resident GPU model.  The sidecar owns its own VRAM lifecycle (lazy load +
# POST /unload).
# ---------------------------------------------------------------------------
_REMOTE_MODELS = {"unlimited-ocr"}

# All model names the server recognises
_ALL_MODELS = set(SUPPORTED_MODELS.keys())

# ---------------------------------------------------------------------------
# Pydantic request / response models (OpenAI Chat Completions format)
# ---------------------------------------------------------------------------


class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "ocr_engine"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a ``data:image/...;base64,...`` URL into a PIL Image."""
    if data_url.startswith("data:"):
        # Strip the header: "data:image/png;base64,<payload>"
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def extract_image_and_prompt(
    messages: list[ChatMessage],
) -> tuple[Optional[Image.Image], str]:
    """Extract the first image and concatenated text from OpenAI-style messages."""
    image: Optional[Image.Image] = None
    texts: list[str] = []

    for msg in messages:
        if isinstance(msg.content, str):
            texts.append(msg.content)
            continue
        for part in msg.content:
            if part.type == "text" and part.text:
                texts.append(part.text)
            elif part.type == "image_url" and part.image_url:
                if image is None:
                    image = decode_base64_image(part.image_url.url)

    prompt = "\n".join(texts) if texts else "请识别图片中的所有文字内容，保持原有格式。"
    return image, prompt


# ---------------------------------------------------------------------------
# ModelManager – lazy-loading with GPU exclusion
# ---------------------------------------------------------------------------


class ModelManager:
    """Manage OCR engine instances with GPU exclusion and VRAM safety.

    * At most **one** GPU model is loaded at a time.
    * RapidOCR / Tesseract (CPU-only) can always coexist.
    * VRAM budget check before loading — refuses if insufficient.
    * Enhanced unload: gc.collect() + empty_cache() + ipc_collect().
    * All load/unload operations are serialised via an ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        self._engines: dict[str, OCREngine] = {}
        self._gpu_model: Optional[str] = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _get_free_vram_mb() -> int:
        """Get free GPU VRAM in MB via torch.cuda."""
        try:
            import torch
            free, _ = torch.cuda.mem_get_info(0)
            return int(free / 1024 / 1024)
        except Exception:
            return 0

    @staticmethod
    def _thorough_gpu_cleanup() -> None:
        """Force-release GPU memory as thoroughly as possible."""
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    async def get_engine(self, model_name: str) -> OCREngine:
        """Return a ready-to-use ``OCREngine``, loading on demand."""
        if model_name not in _ALL_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {sorted(_ALL_MODELS)}"
            )

        async with self._lock:
            # Already loaded – return immediately
            if model_name in self._engines:
                return self._engines[model_name]

            # Remote (sidecar) models hold no VRAM here — treat them like CPU
            # models for exclusion/budget purposes.
            is_gpu = model_name not in _CPU_ONLY_MODELS and model_name not in _REMOTE_MODELS

            # If requesting a GPU model and a *different* GPU model is loaded,
            # unload the old one first.
            if is_gpu and self._gpu_model and self._gpu_model != model_name:
                old = self._engines.pop(self._gpu_model)
                logger.info("Unloading GPU model %s …", self._gpu_model)
                await asyncio.to_thread(old.unload)
                self._gpu_model = None
                # Thorough cleanup after unload
                await asyncio.to_thread(self._thorough_gpu_cleanup)
                free_after = self._get_free_vram_mb()
                logger.info("GPU cleanup done. Free VRAM: %d MB", free_after)

            # VRAM budget check for GPU models
            if is_gpu:
                needed_mb = _MODEL_VRAM_MB.get(model_name, 4000)
                free_mb = self._get_free_vram_mb()
                if free_mb < needed_mb + _VRAM_RESERVE_MB:
                    raise ValueError(
                        f"Insufficient VRAM for {model_name}: "
                        f"need ~{needed_mb}+{_VRAM_RESERVE_MB} MB, "
                        f"only {free_mb} MB free. "
                        f"Try a smaller model (glm-ocr, hunyuan-ocr, lighton-ocr) "
                        f"or free GPU memory."
                    )

            # Load the new model (potentially slow – run in a thread)
            logger.info("Loading model %s (est. %d MB) …",
                        model_name,
                        _MODEL_VRAM_MB.get(model_name, 0))
            engine = OCREngine(model_name=model_name)
            await asyncio.to_thread(engine.load)
            self._engines[model_name] = engine

            if is_gpu:
                self._gpu_model = model_name

            free_now = self._get_free_vram_mb()
            logger.info("Model %s ready. Free VRAM: %d MB", model_name, free_now)
            return engine

    @property
    def loaded_models(self) -> list[str]:
        return list(self._engines.keys())

    @property
    def gpu_model(self) -> Optional[str]:
        return self._gpu_model


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="OCR Engine API", version="0.2.0")
manager = ModelManager()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported")

    model_name = request.model
    try:
        engine = await manager.get_engine(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    image, prompt = extract_image_and_prompt(request.messages)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="No image found in messages. Provide an image_url content part.",
        )

    try:
        result = await asyncio.to_thread(engine.ocr, image, prompt)
    except Exception as exc:
        logger.exception("OCR inference failed for model %s", model_name)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    completion_tokens = result.tokens_generated or 0
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChoiceMessage(content=result.text),
            )
        ],
        usage=UsageInfo(
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        ),
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        data=[ModelCard(id=name) for name in sorted(_ALL_MODELS)]
    )


@app.get("/health")
async def health():
    free_mb = ModelManager._get_free_vram_mb()
    return {
        "status": "ok",
        "loaded_models": manager.loaded_models,
        "gpu_model": manager.gpu_model,
        "free_vram_mb": free_mb,
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model: Optional[str] = None,
    log_level: str = "info",
) -> None:
    """Start the OCR Engine API server.

    Args:
        host: Bind address.
        port: Bind port.
        model: Optional model to preload at startup.
        log_level: Uvicorn log level.
    """

    if model:
        @app.on_event("startup")
        async def _preload():
            logger.info("Preloading model %s …", model)
            await manager.get_engine(model)

    uvicorn.run(app, host=host, port=port, log_level=log_level)
