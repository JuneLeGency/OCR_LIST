# GLM-OCR 集成指南

## 概述

GLM-OCR 是智谱 AI 推出的 OCR 模型，现已集成到本项目中。支持两种使用方式：
1. **直接加载模型**（推荐）：与其他模型一致，简单直接
2. **vLLM 服务器部署**：适合生产环境，支持高并发

## 特性

- **多种识别模式**：支持文本识别、文档解析、公式识别、表格识别、信息提取
- **高精度**：基于 GLM 系列模型的强大能力
- **ModelScope 支持**：使用 ModelScope 下载模型
- **灵活部署**：支持直接加载和服务器部署两种方式

## 安装依赖

```bash
# 安装所有依赖（包含 modelscope 和 openai）
uv sync
```

## 使用方式

### 方式一：直接加载模型（推荐）

这是最简单的方式，不需要启动服务器，与项目中其他模型使用方式一致。

#### 1. 测试单张图片

```bash
# 直接运行
uv run python test_glm_ocr_single.py ./inputs/test.jpg

# 指定识别模式
uv run python test_glm_ocr_single.py ./inputs/test.jpg "Document Parse:"
```

#### 2. 批量处理

```bash
# 仅运行 GLM-OCR
uv run python run_all_models_separate.py --models glmocr

# GLM-OCR + RapidOCR 对比
uv run python run_all_models_separate.py --models rapidocr glmocr

# 所有模型对比（包括 GLM-OCR）
uv run python run_all_models_separate.py --models rapidocr qwen3vl hunyuan glmocr

# 限制处理图片数量
uv run python run_all_models_separate.py --models glmocr --limit 10
```

### 方式二：vLLM 服务器部署（可选）

适合需要高并发或独立服务的场景。

#### 1. 启动服务器

```bash
# 使用提供的脚本
python start_glm_ocr_server.py

# 或直接使用 vLLM 命令
export VLLM_USE_MODELSCOPE=true
python -m vllm.entrypoints.openai.api_server \
    --model ZhipuAI/GLM-OCR \
    --served-model-name glm-ocr \
    --host 0.0.0.0 \
    --port 9090 \
    --trust-remote-code \
    --allowed-local-media-path / \
    --max-model-len 8192
```

#### 2. 使用 API 调用

```python
from glm_ocr import process_image_with_glm_ocr_api

result = process_image_with_glm_ocr_api(
    "./inputs/test.jpg",
    prompt="Text Recognition:",
    base_url="http://localhost:9090/v1"
)
```

## 识别模式

GLM-OCR 支持多种识别模式，通过不同的 prompt 激活：

| 模式 | Prompt | 说明 |
|------|--------|------|
| 文本识别 | `Text Recognition:` | 提取图片中的所有文字 |
| 文档解析 | `Document Parse:` | 解析文档结构和内容 |
| 公式识别 | `Formula Recognition:` | 识别数学公式 |
| 表格识别 | `Table Recognition:` | 提取表格数据 |
| 信息提取 | `Information Extraction:` | 智能提取关键信息 |

## Python API 使用

```python
from glm_ocr import process_image_with_glm_ocr

# 基础使用
result = process_image_with_glm_ocr("./inputs/test.jpg")

# 指定识别模式
result = process_image_with_glm_ocr(
    "./inputs/test.jpg",
    prompt="Table Recognition:"
)

# 自定义服务器地址
result = process_image_with_glm_ocr(
    "./inputs/test.jpg",
    base_url="http://192.168.1.100:9090/v1"
)

print(result)
```

## 结果文件

批量处理的结果保存在 `ocr_results/` 目录：

```
ocr_results/
├── glmocr_results_20250205_120000.json  # GLM-OCR 结果
├── rapidocr_results_20250205_120500.json
├── hunyuan_results_20250205_121000.json
└── qwen3vl_results_20250205_121500.json
```

结果格式：
```json
[
  {
    "filename": "25337000000272477597.jpg",
    "amount": 5937.64,
    "text": "识别的完整文本..."
  }
]
```

## 性能参考

- **模型大小**: ~10GB
- **GPU 显存**: 建议 12GB+
- **处理速度**: 取决于模型大小和硬件配置
- **准确率**: 高精度，特别适合中文文档

## 故障排查

### 1. 服务无法启动

**问题**: `ModuleNotFoundError: No module named 'vllm'`

**解决**:
```bash
pip install vllm
```

### 2. 连接失败

**问题**: `Connection refused to localhost:9090`

**解决**:
- 确保 GLM-OCR 服务器已启动
- 检查端口 9090 是否被占用
- 查看服务器启动日志

### 3. GPU 内存不足

**问题**: `CUDA out of memory`

**解决**:
- 使用更小的 `--max-model-len` 参数
- 使用 `--tensor-parallel-size` 进行模型并行
- 使用量化版本的模型

### 4. ModelScope 下载失败

**问题**: 模型下载速度慢或失败

**解决**:
- 设置 ModelScope 镜像
- 使用代理
- 预先下载模型到缓存目录

## 与其他模型对比

| 模型 | 部署方式 | GPU 需求 | 速度 | 特点 |
|------|----------|---------|------|------|
| **GLM-OCR** | vLLM 服务 | 12GB+ | 中等 | 多模式、高精度 |
| **RapidOCR** | 直接调用 | 无 | 快 | 轻量级、CPU 友好 |
| **HunYuan** | transformers | 8GB+ | 中等 | 腾讯出品、精确度高 |
| **Qwen3-VL** | transformers | 4GB+ | 较慢 | 多模态、通用性强 |

## 参考链接

- [GLM-OCR ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR)
- [vLLM 文档](https://docs.vllm.ai/)
- [智谱 AI](https://www.zhipuai.cn/)

## 相关文件

- `glm_ocr.py` - GLM-OCR 核心模块
- `start_glm_ocr_server.py` - 服务器启动脚本
- `test_glm_ocr_single.py` - 单图测试脚本
- `run_all_models_separate.py` - 批量对比脚本
