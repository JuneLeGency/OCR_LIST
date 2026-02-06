# GLM-OCR 快速开始

## 安装依赖

```bash
uv sync
```

## 快速测试

### 1. 单张图片测试

```bash
# 选择一张图片测试
uv run python test_glm_ocr_single.py ./inputs/25337000000272477597.jpg
```

### 2. 批量对比（推荐）

```bash
# GLM-OCR vs RapidOCR（快速对比，10张图片）
uv run python run_all_models_separate.py --models rapidocr glmocr --limit 10
```

### 3. 完整对比（所有模型）

```bash
# 所有四个模型对比
uv run python run_all_models_separate.py --models rapidocr qwen3vl hunyuan glmocr --limit 10
```

## 查看结果

```bash
# 查看结果文件
ls -lh ocr_results/

# 查看 GLM-OCR 结果
cat ocr_results/glmocr_results_*.json
```

## 识别模式

GLM-OCR 支持多种识别模式：

```bash
# 文本识别（默认）
uv run python test_glm_ocr_single.py ./inputs/test.jpg "Text Recognition:"

# 文档解析
uv run python test_glm_ocr_single.py ./inputs/test.jpg "Document Parse:"

# 表格识别
uv run python test_glm_ocr_single.py ./inputs/test.jpg "Table Recognition:"

# 公式识别
uv run python test_glm_ocr_single.py ./inputs/test.jpg "Formula Recognition:"

# 信息提取
uv run python test_glm_ocr_single.py ./inputs/test.jpg "Information Extraction:"
```

## 性能参考

| 模型 | GPU 显存 | 速度 | 准确率 |
|------|---------|------|--------|
| RapidOCR | 无需 GPU | 极快 | 高 |
| GLM-OCR | ~12GB | 中等 | 极高 |
| HunYuan | ~8GB | 中等 | 高 |
| Qwen3-VL | ~4GB | 较慢 | 中等 |

## 完整文档

详细说明请查看：[GLM_OCR_GUIDE.md](./GLM_OCR_GUIDE.md)
