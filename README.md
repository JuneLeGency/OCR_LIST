# OCR Engine

Unified OCR Engine for VLM models.

## Supported Models

- **qwen3-vl**: Qwen/Qwen3-VL-2B-Instruct
- **glm-ocr**: ZhipuAI/GLM-OCR
- **hunyuan-ocr**: Tencent-Hunyuan/HunyuanOCR
- **deepseek-ocr**: deepseek-ai/DeepSeek-OCR-2

## Installation

```bash
uv sync
```

## Usage

### CLI

```bash
# OCR with default model (qwen3-vl)
uv run ocr image.png

# OCR with specific model
uv run ocr -m glm-ocr image.png

# Batch OCR
uv run ocr -m deepseek-ocr *.png

# Custom prompt
uv run ocr -p "Extract table content" invoice.jpg

# List available models
uv run ocr --list-models
```

### Python API

```python
from ocr_engine import OCREngine

# Initialize and load model
engine = OCREngine("qwen3-vl")
engine.load()

# Perform OCR
text = engine.ocr("image.png")
print(text)

# Custom prompt
text = engine.ocr("invoice.jpg", prompt="Extract all text from this invoice")

# Batch OCR
texts = engine.batch_ocr(["img1.png", "img2.png"])
```

## Model Download

Models are cached in `~/.cache/modelscope/hub/models/`. Use ModelScope to download:

```python
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3-VL-2B-Instruct")
snapshot_download("ZhipuAI/GLM-OCR")
snapshot_download("Tencent-Hunyuan/HunyuanOCR")
snapshot_download("deepseek-ai/DeepSeek-OCR-2")
```
