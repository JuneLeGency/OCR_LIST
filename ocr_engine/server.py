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

logger = logging.getLogger("ocr_engine.server")

# ---------------------------------------------------------------------------
# Models that only use CPU (can coexist with any GPU model)
# ---------------------------------------------------------------------------
_CPU_ONLY_MODELS = {"rapidocr"}

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
    """Manage OCR engine instances with GPU exclusion.

    * At most **one** GPU model is loaded at a time.
    * RapidOCR (CPU-only) can always coexist with a GPU model.
    * All load/unload operations are serialised via an ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        self._engines: dict[str, OCREngine] = {}
        self._gpu_model: Optional[str] = None
        self._lock = asyncio.Lock()

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

            is_gpu = model_name not in _CPU_ONLY_MODELS

            # If requesting a GPU model and a *different* GPU model is loaded,
            # unload the old one first.
            if is_gpu and self._gpu_model and self._gpu_model != model_name:
                old = self._engines.pop(self._gpu_model)
                logger.info("Unloading GPU model %s", self._gpu_model)
                await asyncio.to_thread(old.unload)
                self._gpu_model = None

            # Load the new model (potentially slow – run in a thread)
            logger.info("Loading model %s …", model_name)
            engine = OCREngine(model_name=model_name)
            await asyncio.to_thread(engine.load)
            self._engines[model_name] = engine

            if is_gpu:
                self._gpu_model = model_name

            logger.info("Model %s ready", model_name)
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
    return {
        "status": "ok",
        "loaded_models": manager.loaded_models,
        "gpu_model": manager.gpu_model,
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
