# OCR Invoice Processing 使用指南

这个项目整合了多个OCR模型，用于批量处理发票图片，提取金额并对比各模型的准确度。

## 支持的OCR模型

1. **RapidOCR** - 轻量级OCR引擎，速度快
2. **Qwen3-VL** - 阿里巴巴Qwen的视觉语言模型
3. **HunYuan OCR** - 腾讯混元OCR模型

## 项目结构

```
ocr_tests/
├── inputs/                          # 发票图片目录
├── ocr_results/                     # 输出结果目录（自动创建）
├── invoice_ocr_processor.py         # 主处理脚本
├── evaluate_ocr_accuracy.py         # 准确度评估脚本
├── run_ocr_pipeline.py              # 一键启动脚本
├── ground_truth.json                # 手动标注的真实金额（用于评估）
└── OCR_PROCESSING_GUIDE.md          # 本文档
```

## 快速开始

### 1. 准备发票图片

将所有发票图片放入 `./inputs/` 目录，支持的格式：
- JPG / JPEG
- PNG

### 2. 运行OCR处理

**方式一：使用一键启动脚本**

```bash
# 运行处理 + 评估
python run_ocr_pipeline.py --all

# 仅运行处理
python run_ocr_pipeline.py --process

# 仅运行评估
python run_ocr_pipeline.py --evaluate
```

**方式二：直接运行处理脚本**

```bash
python invoice_ocr_processor.py
```

### 3. 准备评估数据（可选）

如果要对比OCR结果的准确度，需要手动标注真实金额：

编辑 `ground_truth.json`，按照以下格式添加数据：

```json
{
  "收费票据 0607575447.jpg": 123.45,
  "收费票据 0607575461.jpg": 456.78,
  "收费票据 0607651172.jpg": 789.00
}
```

### 4. 运行准确度评估

```bash
python evaluate_ocr_accuracy.py
```

## 输出说明

### 处理结果 (`ocr_results/results_*.json`)

结构示例：

```json
[
  {
    "filename": "发票.jpg",
    "image_path": "./inputs/发票.jpg",
    "manual_amount": null,
    "results": {
      "rapidocr": {
        "raw_text": "...",
        "extracted_amount": 123.45,
        "error": null
      },
      "qwen3_vl": {
        "raw_text": "...",
        "extracted_amount": 123.45,
        "error": null
      },
      "hunyuan": {
        "raw_text": "...",
        "extracted_amount": 123.45,
        "error": null
      }
    }
  }
]
```

### 评估报告 (`ocr_results/accuracy_report_*.json`)

显示每个模型的准确度指标：
- **Accuracy**: 完全正确的百分比
- **Avg Error**: 平均错误金额
- **Max Error**: 最大错误金额

## 金额提取规则

脚本支持多种中文发票金额格式：

- 合计：123.45
- 总计：123.45
- 金额：123.45 / 金额 123.45
- ￥123.45 / $123.45
- 123.45元
- 123.45 RMB
- 应付：123.45
- 应交：123.45

## 模型选择说明

| 模型 | 优势 | 劣势 |
|------|------|------|
| RapidOCR | 速度快、无GPU需求 | 准确度相对较低 |
| Qwen3-VL | 通用性强、理解能力好 | 可能过度解释 |
| HunYuan | OCR专用、准确度高 | 速度较慢、显存需求高 |

## 常见问题

### Q: 运行时提示模型加载失败？

A: 确保已安装必要的依赖包。部分模型首次使用会自动下载（需网络连接）。

### Q: 如何加速处理？

A:
- 使用RapidOCR（不需要GPU）
- 可以在`invoice_ocr_processor.py`中注释掉某些模型的初始化

### Q: 为什么某个模型提取不到金额？

A: 可能原因：
- 发票图片清晰度不足
- 金额格式不符合预设规则
- 可以在`extract_amount_from_text()`方法中添加新的正则表达式

## 自定义和扩展

### 添加新的金额提取规则

编辑`invoice_ocr_processor.py`中的`extract_amount_from_text()`方法：

```python
patterns = [
    r'你的正则表达式',
    # ... 添加新规则
]
```

### 集成新的OCR模型

在`_init_models()`和对应的`run_xxx()`方法中添加新模型。

### 自定义评估指标

编辑`evaluate_ocr_accuracy.py`中的`AccuracyMetrics`类。

## 输出示例

```
================================================================================
INVOICE OCR PROCESSING REPORT
================================================================================
Generated: 2024-01-15 10:30:45

Total invoices processed: 3
Models used: hunyuan, qwen3_vl, rapidocr

--------------------------------------------------------------------------------
Invoice                                            | RapidOCR     | Qwen3-VL     | HunYuan
--------------------------------------------------------------------------------
收费票据 0607575447.jpg                             | 100.5        | 100.5        | 100.5
收费票据 0607575461.jpg                             | 200.0        | 200.0        | 200.0
...

================================================================================
TOTAL AMOUNTS BY MODEL:
================================================================================
rapidocr             : ¥1,234.56
qwen3_vl             : ¥1,234.56
hunyuan              : ¥1,234.56
```

## 性能参考

- RapidOCR: 单张图片 <1秒
- Qwen3-VL: 单张图片 2-3秒（需GPU）
- HunYuan: 单张图片 3-5秒（需GPU）

## 许可证

根据各模型的许可证要求使用。

## 支持

有问题或建议，欢迎提出！
