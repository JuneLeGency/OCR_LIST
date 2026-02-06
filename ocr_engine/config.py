"""Model configurations for OCR engines."""

from dataclasses import dataclass, field
from typing import Optional

from .base import ModelConfig, InferenceMode


# Default model configurations
# All models are Qwen-based but have different transformers requirements:
#   - qwen3-vl:    Works with transformers main (5.x)
#   - glm-ocr:     Works with transformers main (5.x)
#   - hunyuan-ocr: Requires merged transformers (main + 82a06db patch)
#   - deepseek-ocr: Requires trust_remote_code

SUPPORTED_MODELS = {
    "qwen3-vl": ModelConfig(
        name="Qwen3-VL-2B",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=False,
        torch_dtype="auto",
        max_new_tokens=8192,
    ),
    "glm-ocr": ModelConfig(
        name="GLM-OCR",
        model_id="ZhipuAI/GLM-OCR",
        trust_remote_code=False,
        torch_dtype="auto",
        max_new_tokens=8192,
    ),
    "hunyuan-ocr": ModelConfig(
        name="HunyuanOCR",
        model_id="Tencent-Hunyuan/HunyuanOCR",
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="eager",  # flash_attention_2 incompatible with xdrope
        max_new_tokens=8192,
    ),
    "deepseek-ocr": ModelConfig(
        name="DeepSeek-OCR-2",
        model_id="deepseek-ai/DeepSeek-OCR-2",
        trust_remote_code=True,
        torch_dtype="bfloat16",
        max_new_tokens=8192,
    ),
    "rapidocr": ModelConfig(
        name="RapidOCR",
        model_id="rapidocr",
    ),
}

# vLLM server configurations (examples)
VLLM_CONFIGS = {
    "qwen3-vl-vllm": ModelConfig(
        name="Qwen3-VL-2B (vLLM)",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        vllm_base_url="http://localhost:8000/v1",
        max_new_tokens=8192,
    ),
    "glm-ocr-vllm": ModelConfig(
        name="GLM-OCR (vLLM)",
        model_id="ZhipuAI/GLM-OCR",
        vllm_base_url="http://localhost:9090/v1",
        max_new_tokens=8192,
    ),
    "hunyuan-ocr-vllm": ModelConfig(
        name="HunyuanOCR (vLLM)",
        model_id="tencent/HunyuanOCR",  # HuggingFace model name for vLLM
        vllm_base_url="http://localhost:8001/v1",
        max_new_tokens=8192,
    ),
    "deepseek-ocr-vllm": ModelConfig(
        name="DeepSeek-OCR-2 (vLLM)",
        model_id="deepseek-ai/DeepSeek-OCR-2",
        vllm_base_url="http://localhost:8002/v1",
        max_new_tokens=8192,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    # Check offline models first
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]

    # Check vLLM configs
    if model_name in VLLM_CONFIGS:
        return VLLM_CONFIGS[model_name]

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Offline models: {list(SUPPORTED_MODELS.keys())}. "
        f"vLLM models: {list(VLLM_CONFIGS.keys())}"
    )


def get_backend_name(model_name: str) -> str:
    """Get backend name for a model."""
    if model_name in VLLM_CONFIGS or model_name.endswith("-vllm"):
        return "vllm"
    return model_name


def list_all_models() -> dict[str, list[str]]:
    """List all available models grouped by type."""
    return {
        "offline": list(SUPPORTED_MODELS.keys()),
        "vllm": list(VLLM_CONFIGS.keys()),
    }


def create_vllm_config(
    model_id: str,
    base_url: str,
    api_key: str = "EMPTY",
    **kwargs,
) -> ModelConfig:
    """Create a custom vLLM configuration."""
    return ModelConfig(
        name=f"{model_id} (vLLM)",
        model_id=model_id,
        vllm_base_url=base_url,
        vllm_api_key=api_key,
        **kwargs,
    )
