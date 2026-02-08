# OCR Engine

Unified OCR Engine supporting 8 backends (5 GPU VLM + 3 traditional/lightweight).

## Supported Models

### Traditional OCR (CPU, no GPU required)

| Model | Engine | Model Size | Avg Speed |
|-------|--------|-----------|-----------|
| **rapidocr** | PaddleOCR v4 (ONNX) | 16 MB | ~1.3s/img |
| **tesseract** | Google Tesseract 5 | 19 MB | ~0.7s/img |

### GPU VLM Models

| Model | Developer | Params | Model Size | VRAM (load/peak) | Avg Speed |
|-------|-----------|--------|-----------|------------------|-----------|
| **glm-ocr** | ZhipuAI | 1.3B | 2.5 GB | 2.0 / 3.9 GB | ~2.8s/img |
| **qwen3-vl** | Alibaba | 2.1B | 4.0 GB | 4.1 / 6.2 GB | ~4.9s/img |
| **hunyuan-ocr** | Tencent | 1.0B | 1.9 GB | 2.9 / >32 GB* | ~67s/img |
| **deepseek-ocr** | DeepSeek | 3.4B | 6.4 GB | 6.5 / 8.5 GB | ~2.8s/img |
| **chandra-ocr** | Datalab | 8.9B | 16.5 GB | 16.6 / 18.4 GB | ~23s/img |
| **dots-ocr** | RedNote | 1.7B | 5.7 GB | 5.7 / 6.7 GB | ~14s/img |

### Benchmark Accuracy (47 invoices, majority-vote consensus)

| Model | Accuracy | Match | Mismatch | Error |
|-------|----------|-------|----------|-------|
| glm-ocr | **100.0%** | 47/47 | 0 | 0 |
| qwen3-vl | **100.0%** | 47/47 | 0 | 0 |
| rapidocr | 97.7% | 43/47 | 1 | 3 |
| hunyuan-ocr | 88.9% | 16/47 | 2 | 29 |
| deepseek-ocr | 79.2% | 19/47 | 5 | 23 |

> Tested on RTX 5090 (32 GB). All models BF16 precision. Speed is median per-image time.

## Installation

```bash
# Core (GPU models)
uv sync

# With RapidOCR (CPU)
uv sync --extra rapidocr

# With Tesseract (requires system: apt install tesseract-ocr tesseract-ocr-chi-sim)
uv sync --extra tesseract

# With Chandra
uv sync --extra chandra

# All extras
uv sync --extra rapidocr --extra tesseract --extra chandra
```

## Usage

### CLI

```bash
# OCR with default model (qwen3-vl)
uv run ocr image.png

# OCR with specific model
uv run ocr -m glm-ocr image.png

# Batch OCR
uv run ocr -m rapidocr *.png

# Custom prompt
uv run ocr -p "Extract table content" invoice.jpg

# List available models
uv run ocr --list-models
```

### Python API

```python
from ocr_engine import OCREngine

# Initialize and load model
engine = OCREngine("glm-ocr")
engine.load()

# Perform OCR
result = engine.ocr("image.png")
print(result.text)

# Context manager (auto cleanup)
with OCREngine("rapidocr") as engine:
    result = engine.ocr("invoice.jpg")

# Batch OCR
results = engine.batch_ocr(["img1.png", "img2.png"])
```

### Benchmark

```bash
# Run all models + evaluate
uv run python benchmark.py

# Specific models only
uv run python benchmark.py --models rapidocr glm-ocr qwen3-vl

# Skip models with existing results
uv run python benchmark.py --skip-existing

# Evaluate from existing results only
uv run python benchmark.py --evaluate-only
```

## Model Download

GPU models use ModelScope cache (`~/.cache/modelscope/hub/models/`):

```bash
python setup_modelscope_models.py --download  # Download all models

# Apply required patches
python scripts/patch_deepseek_ocr.py
python scripts/patch_dots_ocr.py
```

## Model Compatibility (transformers 5.x)

This project uses a [custom transformers fork](https://github.com/JuneLeGency/transformers_hy.git)
(branch `hunyuan-vl-patch`) for HunyuanOCR support. All other models work with this fork.

Models requiring post-download patches (applied to ModelScope cache):

| Model | Patch Script | Key Fixes |
|-------|-------------|-----------|
| **deepseek-ocr** | `scripts/patch_deepseek_ocr.py` | inv_freq recompute, LlamaAttention standalone impl, rope_scaling compat, cache API migration |
| **dots-ocr** | `scripts/patch_dots_ocr.py` | Custom from_pretrained (bypass meta device), GenerationConfig loading, DotsVLProcessor kwargs fix |

Models working out-of-the-box (no patches needed):
- **qwen3-vl**, **glm-ocr** — native transformers 5.x support
- **hunyuan-ocr** — fixed in transformers fork
- **chandra-ocr** — HuggingFace auto-download, standard loading
- **rapidocr**, **tesseract** — CPU backends, no transformers dependency
