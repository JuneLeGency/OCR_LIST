# OCR VLM 模型信息整理

用于重构的开源 OCR 视觉语言模型参考文档。

---

## 模型概览

| 模型 | 开发者 | 参数量 | 发布时间 | 特点 |
|------|--------|--------|----------|------|
| Qwen2.5-VL | 阿里云 Qwen | 3B/7B/32B/72B | 2025.01 | OCR能力强，多语言，agentic |
| Qwen3-VL | 阿里云 Qwen | 2B/8B/32B | 2025.10 | 支持32种语言OCR，支持模糊低光 |
| GLM-4V-9B | 智谱AI/THUDM | 9B | 2024 | 中英双语，高分辨率1120x1120 |
| GLM-OCR | 智谱AI | ~2B | 2025 | 专门OCR优化 |
| HunyuanOCR | 腾讯 | 1B | 2025.11 | 轻量SOTA，端到端OCR专家模型 |
| DeepSeek-OCR-2 | DeepSeek | 3B | 2026.01 | Visual Causal Flow，强视觉推理 |

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

## 依赖库对比

| 模型 | 加载库 | Processor | Model Class |
|------|--------|-----------|-------------|
| Qwen3-VL | transformers | AutoProcessor | Qwen3VLForConditionalGeneration |
| GLM-OCR | modelscope | AutoProcessor | AutoModelForImageTextToText |
| HunyuanOCR | transformers | AutoProcessor | HunYuanVLForConditionalGeneration |
| DeepSeek-OCR-2 | transformers | AutoTokenizer | AutoModel (trust_remote_code) |

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

## 重构建议

基于以上分析，可以抽象出统一的接口：

1. **统一的 Model Loader**: 根据模型类型自动选择正确的加载方式
2. **统一的消息格式转换**: 将通用格式转换为各模型特定格式
3. **统一的推理接口**: `process_image(image_path, prompt) -> str`
4. **统一的后处理**: 各模型输出的标准化

```python
# 目标接口示例
from ocr_engine import OCREngine

engine = OCREngine(model="hunyuan")  # 或 "qwen", "glm", "deepseek"
result = engine.process("image.jpg", prompt="识别文字")
```
