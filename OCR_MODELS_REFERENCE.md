# OCR VLM 模型信息整理

用于重构的开源 OCR 视觉语言模型参考文档。

---

## 模型概览

| 模型 | 开发者 | 参数量 | 发布时间 | 特点 |
|------|--------|--------|----------|------|
| Qwen2.5-VL | 阿里云 Qwen | 3B/7B/32B/72B | 2025.01 | OCR能力强，多语言，agentic |
| Qwen3-VL | 阿里云 Qwen | 2B/8B/32B | 2025.10 | 支持32种语言OCR，支持模糊低光 |
| GLM-4V-9B | 智谱AI/THUDM | 9B | 2024 | 中英双语，高分辨率1120x1120 |
| GLM-OCR | 智谱AI | 1.3B | 2025 | 专门OCR优化，发票场景精度极高 |
| HunyuanOCR | 腾讯 | 1.0B | 2025.11 | 轻量SOTA，端到端OCR专家模型 |
| DeepSeek-OCR-2 | DeepSeek | 3.4B | 2026.01 | Visual Causal Flow，强视觉推理 |
| Chandra-OCR | Datalab | 8.9B | 2025 | 文档布局解析+OCR，Markdown输出 |
| dots.ocr | RedNote (小红书) | 3.1B | 2025 | 多语言文档布局解析VLM |
| RapidOCR | PaddleOCR | — | — | 传统OCR，CPU推理，ONNX加速 |
| Tesseract | Google | — | — | 传统OCR，CPU推理，开源经典 |

### 技术参数对比

| 模型 | 模型大小 | 精度 | 显存(加载/峰值) | 推理速度 | 推理方式 |
|------|---------|------|-----------------|---------|---------|
| rapidocr | 16 MB | FP32 | — (CPU) | ~1.3s/img | ONNX Runtime |
| tesseract | 19 MB | — | — (CPU) | ~0.7s/img | C++ Engine |
| glm-ocr | 2.5 GB | BF16 | 2.0 / 3.9 GB | ~2.8s/img | transformers |
| qwen3-vl | 4.0 GB | BF16 | 4.1 / 6.2 GB | ~4.9s/img | transformers |
| hunyuan-ocr | 1.9 GB | BF16 | 2.9 / >32 GB* | ~67s/img | transformers (eager attn) |
| deepseek-ocr | 6.4 GB | BF16 | 6.5 / 8.5 GB | ~2.8s/img | transformers |
| chandra-ocr | 16.5 GB | BF16 | 16.6 / 18.4 GB | ~23s/img | chandra SDK |
| dots-ocr | 5.7 GB | BF16 | 22.5 / >32 GB* | OOM | transformers |

\* eager attention 导致高分辨率图片显存暴增；32 GB 显存不足以处理部分输入。

> 测试环境: RTX 5090 (32 GB), Python 3.12, torch 2.9, transformers 5.x

### 基准评测（47 张发票，Majority-Vote 共识）

| 模型 | 准确率 | 匹配 | 不匹配 | 错误 |
|------|--------|------|--------|------|
| glm-ocr | **100.0%** | 47/47 | 0 | 0 |
| qwen3-vl | **100.0%** | 47/47 | 0 | 0 |
| rapidocr | 97.7% | 43/47 | 1 | 3 |
| hunyuan-ocr | 88.9% | 16/47 | 2 | 29 |
| deepseek-ocr | 79.2% | 19/47 | 5 | 23 |

> 评测方式：所有模型对同一张图提取金额，取多数一致的结果作为标准答案。accuracy = match / (match + mismatch)。

---

## 1. Qwen VL 系列

### Qwen2.5-VL

**特点：**
- OCR能力升级，多场景、多语言、多方向文字识别
- 支持 Agentic 能力（电脑/手机操控）
- 支持超过1小时的视频理解

**HuggingFace：**
- Collection: [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl)
- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)

**ModelScope (魔搭)：**
- Collection: [Qwen2.5-VL](https://modelscope.cn/collections/Qwen25-VL-58fbb5d31f1d47)
- [Qwen/Qwen2.5-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2.5-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct)

**GitHub：** [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

### Qwen3-VL (最新)

**特点：**
- 支持 32 种语言的 OCR（从10种扩展）
- 低光、模糊、倾斜条件下表现更好
- 稀有/古代字符和术语识别更佳
- 长文档结构解析改进

**HuggingFace：**
- [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)

**ModelScope (魔搭)：**
- [Qwen/Qwen3-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen/Qwen3-VL-8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)

**GitHub：** [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

**调用方式：**
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "请识别图片中的文字"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)
```

---

## 2. GLM 系列

### GLM-4V-9B

**特点：**
- 中英双语多轮对话
- 1120x1120 高分辨率支持
- 多模态基准测试优于 GPT-4-turbo、Gemini 1.0 Pro、Claude 3 Opus

**HuggingFace：**
- [THUDM/glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b)
- [GLM-4 Collection](https://huggingface.co/collections/THUDM/glm-4-665fcf188c414b03c2f7e3b7)

**ModelScope (魔搭)：**
- [ZhipuAI/glm-4v-9b](https://modelscope.cn/models/ZhipuAI/glm-4v-9b)

**GitHub：** [THUDM/GLM-4](https://github.com/THUDM/GLM-4)

### GLM-OCR (专门OCR优化)

**特点：**
- 专门针对 OCR 任务优化的版本
- 支持 vLLM 部署

**ModelScope (魔搭)：**
- [ZhipuAI/GLM-OCR](https://modelscope.cn/models/ZhipuAI/GLM-OCR)

**调用方式：**
```python
from modelscope import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "ZhipuAI/GLM-OCR"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "path/to/image.jpg"},
            {"type": "text", "text": "Text Recognition:"}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)

generated_ids = model.generate(**inputs, max_new_tokens=8192)
output = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:])
```

---

## 3. HunyuanOCR

**特点：**
- 仅 1B 参数实现 SOTA 性能
- 端到端 OCR 专家模型
- 支持文字检测识别、复杂文档解析、开放域信息提取、视频字幕提取、图片翻译
- 多语言支持
- ICDAR 2025 DIMT 挑战赛小模型赛道第一名

**Benchmark：**
- OCRBench: 860 (3B以下模型最高)
- OmniDocBench: 94.1 (复杂文档解析领先)

**HuggingFace：**
- [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
- [Demo Space](https://huggingface.co/spaces/tencent/HunyuanOCR)

**ModelScope (魔搭)：**
- [Tencent-Hunyuan/HunyuanOCR](https://modelscope.cn/models/Tencent-Hunyuan/HunyuanOCR)

**GitHub：** [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)

**调用方式：**
```python
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from PIL import Image
import torch

model_name = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
model = HunYuanVLForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "检测并识别图片中的文字"},
        ],
    }
]

texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
image = Image.open("path/to/image.jpg")
inputs = processor(text=texts, images=image, padding=True, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)
```

---

## 4. DeepSeek-OCR-2

**特点：**
- 3B 参数模型
- Visual Causal Flow 架构
- 使用 Qwen2-0.5B 作为视觉编码器 (DeepEncoder V2)
- 不仅提取文字，还具有强视觉推理能力

**HuggingFace：**
- [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [unsloth/DeepSeek-OCR-2](https://huggingface.co/unsloth/DeepSeek-OCR-2) (量化版)
- [NexaAI/DeepSeek-OCR-GGUF](https://huggingface.co/NexaAI/DeepSeek-OCR-GGUF) (GGUF版)

**ModelScope (魔搭)：**
- [deepseek-ai/DeepSeek-OCR-2](https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR-2)

**GitHub：** [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)

**调用方式：**
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)

# 基本OCR
prompt = "<image>\n<|grounding|>Convert the document to markdown"
# 使用模型推理...
```

---

## 5. Chandra-OCR

**特点：**
- 8.9B 参数文档理解模型
- 结合 OCR + 文档布局解析，输出结构化 Markdown
- 基于 HuggingFace 模型，支持 `hf` / `vllm` 两种推理方式
- 安装：`uv sync --extra chandra`（独立 pip 包 `chandra-ocr`）

**HuggingFace：**
- [datalab-to/chandra](https://huggingface.co/datalab-to/chandra)

**GitHub：** [datalab-to/chandra](https://github.com/datalab-to/chandra)

**调用方式：**
```python
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from PIL import Image

manager = InferenceManager(method="hf")
image = Image.open("document.jpg")
batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
results = manager.generate(batch)
print(results[0].markdown)
```

---

## 6. Tesseract

**特点：**
- Google 开源经典 OCR 引擎（Tesseract 5）
- 纯 CPU 推理，无需 GPU
- 支持中英文（需安装语言包 `tesseract-ocr-chi-sim`）
- 速度快（~0.7s/img），适合轻量场景
- 安装：`uv sync --extra tesseract` + 系统包 `apt install tesseract-ocr tesseract-ocr-chi-sim`

**GitHub：** [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

**调用方式：**
```python
import pytesseract
from PIL import Image

image = Image.open("document.jpg")
text = pytesseract.image_to_string(image, lang="chi_sim+eng")
print(text)
```

---

## 7. dots.ocr

**特点：**
- 3.1B 参数多语言文档布局解析 VLM（小红书 RedNote 出品）
- 输出 JSON 格式的布局信息（bbox + category + text）
- 支持 Caption, Formula (LaTeX), Table (HTML), Text (Markdown) 等布局类型
- 注意：dots.ocr PyPI 包锁定 `torch==2.7.0`，本项目直接用 `transformers.AutoModelForCausalLM` 加载，无额外依赖
- 已知问题：在 32 GB 显存下 OOM（模型加载 22.5 GB + 推理需要额外 >10 GB）

**HuggingFace：**
- [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)

**GitHub：** [rednote-hilab/dots.ocr1.5](https://github.com/rednote-hilab/dots.ocr1.5)

**调用方式：**
```python
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "rednote-hilab/dots.ocr"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

messages = [{"role": "user", "content": [
    {"type": "image", "image": "document.jpg"},
    {"type": "text", "text": "Please output the layout information..."},
]}]

text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=24000)
# Output is JSON with layout elements, each containing bbox, category, text
```

---

## 8. RapidOCR

**特点：**
- 基于 PaddleOCR v4 的 ONNX 推理方案
- 纯 CPU 推理，轻量高效（模型仅 16 MB）
- 支持中英文混合识别
- 安装：`uv sync --extra rapidocr`

**GitHub：** [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)

**调用方式：**
```python
from rapidocr import RapidOCR

engine = RapidOCR()
result = engine("document.jpg")
# result: list of (bbox, text, confidence)
```

---

## 依赖库对比

| 模型 | 加载库 | Processor | Model Class | trust_remote_code |
|------|--------|-----------|-------------|-------------------|
| Qwen3-VL | transformers | AutoProcessor | Qwen3VLForConditionalGeneration | No |
| GLM-OCR | transformers | AutoProcessor | AutoModelForImageTextToText | No |
| HunyuanOCR | transformers | AutoProcessor | HunYuanVLForConditionalGeneration | Yes |
| DeepSeek-OCR-2 | transformers | AutoTokenizer | AutoModel | Yes |
| Chandra-OCR | chandra SDK | — | InferenceManager | — |
| dots.ocr | transformers | AutoProcessor | AutoModelForCausalLM | Yes |
| RapidOCR | rapidocr | — | RapidOCR | — |
| Tesseract | pytesseract | — | — | — |

---

## 消息格式统一

所有模型都采用类似的消息格式：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image"},  # 或 "url"
            {"type": "text", "text": "OCR prompt"},
        ],
    }
]
```

**差异点：**
- GLM-OCR: 使用 `"url"` 而不是 `"image"` 作为图片路径的 key
- HunyuanOCR: 需要额外的 system message
- DeepSeek-OCR-2: 使用特殊的 grounding token

---

## 统一接口（已实现）

所有模型已通过 `ocr_engine` 统一封装：

```python
from ocr_engine import OCREngine

# 任意模型使用相同接口
engine = OCREngine("glm-ocr")  # 或 qwen3-vl, rapidocr, tesseract, ...
engine.load()
result = engine.ocr("image.jpg")
print(result.text)
engine.unload()

# Context manager 自动加载/卸载
with OCREngine("rapidocr") as engine:
    result = engine.ocr("invoice.jpg")
```

支持的模型名：`qwen3-vl`, `glm-ocr`, `hunyuan-ocr`, `deepseek-ocr`, `chandra-ocr`, `dots-ocr`, `rapidocr`, `tesseract`
