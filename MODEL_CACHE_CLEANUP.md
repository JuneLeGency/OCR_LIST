# OCR 模型缓存整理

统一使用 ModelScope 避免重复下载和网络问题。

---

## 当前缓存状态 (已清理后)

### ModelScope (`~/.cache/modelscope/hub/models/`)
| 模型 | 路径 | 大小 | 状态 |
|------|------|------|------|
| GLM-OCR | ZhipuAI/GLM-OCR | 2.5G | ✅ 完整 |
| Qwen3-VL-2B | Qwen/Qwen3-VL-2B-Instruct | ~4G | ❌ 待下载 |
| HunyuanOCR | Tencent-Hunyuan/HunyuanOCR | ~2G | ❌ 待下载 |
| DeepSeek-OCR-2 | deepseek-ai/DeepSeek-OCR-2 | ~6G | ❌ 待下载 |

### HuggingFace (`~/.cache/huggingface/hub/`) - 待清理
| 模型 | 路径 | 大小 | 状态 |
|------|------|------|------|
| Qwen3-VL-2B | models--Qwen--Qwen3-VL-2B-Instruct | 7.9G | ⚠️ ModelScope下载后可删除 |
| HunyuanOCR | models--tencent--HunyuanOCR | 7.4G | ⚠️ ModelScope下载后可删除 |
| ~~HunYuanOCR~~ | ~~models--tencent--HunYuanOCR~~ | - | ✅ 已删除 (6.2G释放) |
| ~~GLM-OCR~~ | ~~models--zai-org--GLM-OCR~~ | - | ✅ 已删除 |
| ~~chandra~~ | ~~models--datalab-to--chandra~~ | - | ✅ 已删除 |

---

## ModelScope 模型路径对照表

| 模型 | HuggingFace 路径 | ModelScope 路径 | 备注 |
|------|-----------------|-----------------|------|
| Qwen3-VL-2B | Qwen/Qwen3-VL-2B-Instruct | Qwen/Qwen3-VL-2B-Instruct | 路径相同 |
| GLM-OCR | zai-org/GLM-OCR | ZhipuAI/GLM-OCR | ⚠️ 路径不同 |
| HunyuanOCR | tencent/HunyuanOCR | Tencent-Hunyuan/HunyuanOCR | ⚠️ 路径不同 |
| DeepSeek-OCR-2 | deepseek-ai/DeepSeek-OCR-2 | deepseek-ai/DeepSeek-OCR-2 | 路径相同 |

---

## 清理步骤

### 步骤 1: 删除不需要的 HuggingFace 缓存

```bash
# 删除重复的 HunYuanOCR (大小写问题)
rm -rf ~/.cache/huggingface/hub/models--tencent--HunYuanOCR

# 删除不完整的 GLM-OCR
rm -rf ~/.cache/huggingface/hub/models--zai-org--GLM-OCR

# 删除不相关的模型
rm -rf ~/.cache/huggingface/hub/models--datalab-to--chandra
```

### 步骤 2: 从 ModelScope 下载模型

```python
from modelscope import snapshot_download
import os

os.environ["MODELSCOPE_CACHE"] = os.path.expanduser("~/.cache/modelscope")

# 下载 Qwen3-VL-2B
snapshot_download('Qwen/Qwen3-VL-2B-Instruct')

# 下载 HunyuanOCR
snapshot_download('Tencent-Hunyuan/HunyuanOCR')

# 下载 DeepSeek-OCR-2 (尚未下载)
snapshot_download('deepseek-ai/DeepSeek-OCR-2')
```

### 步骤 3: 删除原 HuggingFace 缓存 (下载完成后)

```bash
# 确认 ModelScope 下载完成后删除
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct
rm -rf ~/.cache/huggingface/hub/models--tencent--HunyuanOCR
```

---

## 最终目标目录结构

```
~/.cache/modelscope/hub/models/
├── ZhipuAI/
│   └── GLM-OCR/              # 2.5G (已有)
├── Qwen/
│   └── Qwen3-VL-2B-Instruct/ # ~4G (待下载)
├── Tencent-Hunyuan/
│   └── HunyuanOCR/           # ~2G (待下载)
└── deepseek-ai/
    └── DeepSeek-OCR-2/       # ~6G (待下载)
```

---

## 代码中统一使用 ModelScope

所有模型加载代码改为使用 modelscope:

```python
import os
os.environ["VLLM_USE_MODELSCOPE"] = "true"

# GLM-OCR
from modelscope import AutoProcessor, AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("ZhipuAI/GLM-OCR")

# Qwen3-VL
from modelscope import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# HunyuanOCR
from modelscope import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Tencent-Hunyuan/HunyuanOCR")

# DeepSeek-OCR-2
from modelscope import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR-2")
```

---

## 预计磁盘空间

| 操作 | 变化 |
|------|------|
| 删除 HuggingFace 重复/无用 | -2.0G |
| 下载 Qwen3-VL (ModelScope) | +4.0G |
| 下载 HunyuanOCR (ModelScope) | +2.0G |
| 删除 HuggingFace Qwen3-VL | -4.0G |
| 删除 HuggingFace HunyuanOCR | -1.9G |
| 下载 DeepSeek-OCR-2 (新) | +6.0G |
| **净增加** | **+4.1G** |
