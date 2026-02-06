# 发票 OCR 完整使用指南 | Complete Invoice OCR Usage Guide

---

## 目录 | Table of Contents

1. [快速开始](#快速开始)
2. [详细说明](#详细说明)
3. [模型选择](#模型选择)
4. [故障排除](#故障排除)
5. [性能对比](#性能对比)

---

## 快速开始

### 场景 1️⃣: 我只想快速处理发票

```bash
# 一行命令搞定！
uv run python rapidocr_batch_processor.py

# 结果在这里查看
cat ocr_results/summary_*.txt
```

**耗时**: ~60 秒处理 47 张发票
**成功率**: 100%
**GPU需求**: ❌ 无需

---

### 场景 2️⃣: 我想对比 RapidOCR 和 HunYuan

```bash
# 运行对比
uv run python run_all_models_separate.py \
  --models rapidocr hunyuan \
  --limit 10

# 查看详细结果
cat FINAL_MODEL_COMPARISON.md
```

**耗时**: ~2 分钟处理 10 张发票
**GPU需求**: ✅ 需要 8GB+ VRAM

---

### 场景 3️⃣: 我想测试所有模型

```bash
# 全部运行（会很慢）
uv run python run_all_models_separate.py \
  --models rapidocr qwen3vl hunyuan \
  --limit 10

# 监控进度（在另一个终端）
watch 'ls -lh ocr_results/ | tail -5'
```

**耗时**: ~15 分钟处理 10 张发票
**GPU需求**: ✅ 需要 8GB+ VRAM
**预期时间**: Qwen3-VL 较慢（5-7秒/张）

---

## 详细说明

### RapidOCR 处理流程

#### 1. 检查输入

```bash
# 查看有多少张发票
ls -lh inputs/ | wc -l

# 查看发票列表
ls inputs/ | head -10
```

#### 2. 运行处理

```bash
# 基本运行
uv run python rapidocr_batch_processor.py

# 或者直接用 Python
python rapidocr_batch_processor.py
```

#### 3. 查看结果

```bash
# 方式 1: 查看摘要
cat ocr_results/summary_*.txt

# 方式 2: 查看完整 JSON
cat ocr_results/rapidocr_results_*.json | python -m json.tool | head -100

# 方式 3: 用 Python 统计
python3 << 'EOF'
import json
from pathlib import Path

# 找最新的结果文件
result_files = sorted(Path('ocr_results').glob('rapidocr_results_*.json'))
if result_files:
    with open(result_files[-1]) as f:
        results = json.load(f)

    # 统计
    total = len(results)
    success = sum(1 for r in results if r.get('amount'))
    total_amount = sum(r['amount'] for r in results if r.get('amount'))

    print(f"总发票数: {total}")
    print(f"成功提取: {success}/{total}")
    print(f"总金额: ¥{total_amount:.2f}")

    # 显示前 5 张
    print("\n前 5 张发票:")
    for r in results[:5]:
        amount = f"¥{r['amount']:.2f}" if r.get('amount') else "N/A"
        print(f"  {r['filename']:<40} {amount}")
EOF
```

### HunYuan 处理流程

#### 1. 检查 GPU

```bash
# 查看 GPU 信息
nvidia-smi

# 需要至少 8GB 空闲显存
# 如果显存不足，清理一下：
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### 2. 运行 HunYuan

```bash
# 方式 1: 使用分离脚本（推荐）
uv run python run_all_models_separate.py --models hunyuan --limit 10

# 方式 2: 直接运行简单测试
uv run python test_hunyuan_simple.py

# 方式 3: 运行改进的对比脚本
uv run python compare_models_fixed.py --limit 10
```

#### 3. 查看 HunYuan 结果

```bash
# 查看最新结果
cat ocr_results/hunyuan_results_*.json | python -m json.tool | head -100

# 用 Python 统计
python3 << 'EOF'
import json
from pathlib import Path

result_files = sorted(Path('ocr_results').glob('hunyuan_results_*.json'))
if result_files:
    with open(result_files[-1]) as f:
        results = json.load(f)

    success = sum(1 for r in results if r.get('amount'))
    failed = sum(1 for r in results if 'error' in r or not r.get('amount'))
    total_amount = sum(r['amount'] for r in results if r.get('amount'))

    print(f"总发票数: {len(results)}")
    print(f"成功: {success}, 失败: {failed}")
    print(f"总金额: ¥{total_amount:.2f}")
EOF
```

### 对比结果

```bash
# 生成对比报告
python3 << 'EOF'
import json
from pathlib import Path

# 加载两个结果
rapid_files = sorted(Path('ocr_results').glob('rapidocr_results_*.json'))
hunyuan_files = sorted(Path('ocr_results').glob('hunyuan_results_*.json'))

if rapid_files and hunyuan_files:
    with open(rapid_files[-1]) as f:
        rapid = json.load(f)
    with open(hunyuan_files[-1]) as f:
        hunyuan = json.load(f)

    print("=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    matches = 0
    for hy in hunyuan:
        ro = next((r for r in rapid if r['filename'] == hy['filename']), None)
        if not ro:
            continue

        ro_amount = ro.get('extracted_amount')
        hy_amount = hy.get('amount')

        if ro_amount and hy_amount and abs(ro_amount - hy_amount) < 0.01:
            match = "✓"
            matches += 1
        elif not ro_amount or not hy_amount:
            match = "✗ (Missing)"
        else:
            match = "✗"

        print(f"{hy['filename']:<45} {match}")

    print("-" * 80)
    print(f"Matching: {matches}/{len(hunyuan)}")
EOF
```

---

## 模型选择

### 对比表

| 特性 | RapidOCR | HunYuan | Qwen3-VL |
|------|----------|---------|----------|
| 速度 | ⚡⚡⚡ 快 | 🐢 慢 | 🚶 中 |
| 准确度 | ⭐⭐⭐ 高 | ⭐⭐ 中 | ⭐ 低 |
| 成功率 | 100% | 80% | 70% |
| GPU需求 | ❌ 无 | ✅ 8-12GB | ✅ 4-8GB |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| 金额提取 | ✓ 优秀 | ✓ 优秀 | ✗ 差 |

### 决策流程

```
选择模型
│
├─ 我没有 GPU
│  └─> RapidOCR ✓
│
├─ 我有 GPU，想要快速
│  └─> RapidOCR ✓
│
├─ 我有 GPU，想要高质量
│  └─> RapidOCR + HunYuan ✓
│
├─ 我想对比所有模型
│  └─> run_all_models_separate.py --models rapidocr qwen3vl hunyuan ✓
│
└─ 我想用 Qwen3-VL 做金额提取
   └─> ⚠️ 不推荐！准确度太低
```

---

## 故障排除

### 问题 1: "ModuleNotFoundError: No module named 'rapidocr'"

**解决:**
```bash
# 安装依赖
pip install rapidocr-onnxruntime

# 或使用 uv
uv sync
```

### 问题 2: "CUDA out of memory"

**原因:** 显存不足或在同一进程加载了多个大模型

**解决:**
```bash
# ✗ 错误方式
python compare_models.py  # 会 OOM

# ✓ 正确方式
python run_all_models_separate.py --models hunyuan
```

### 问题 3: "FileNotFoundError: inputs/ not found"

**解决:**
```bash
# 创建输入目录
mkdir -p inputs

# 把发票图片放进去
cp /path/to/invoices/*.jpg inputs/
```

### 问题 4: HunYuan "模型加载失败"

**原因:** 网络问题或 Hugging Face 无法访问

**解决:**
```bash
# 方法 1: 检查网络
ping huggingface.co

# 方法 2: 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 方法 3: 提前下载模型
python3 << 'EOF'
from transformers import HunYuanVLForConditionalGeneration, AutoProcessor
AutoProcessor.from_pretrained("tencent/HunyuanOCR")
HunYuanVLForConditionalGeneration.from_pretrained("tencent/HunyuanOCR")
EOF
```

### 问题 5: "No module named 'torch'"

**解决:**
```bash
# 如果使用 uv
uv sync

# 如果使用 pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题 6: 某个发票无法处理

**可能原因:**
1. 发票图片质量太低
2. 发票格式特殊（HunYuan 的限制）
3. GPU 显存波动

**解决:**
```bash
# 方式 1: 重新处理该张发票
uv run python test_hunyuan_simple.py  # 修改路径后运行

# 方式 2: 使用 RapidOCR 替代
# RapidOCR 通常能处理 HunYuan 失败的情况

# 方式 3: 检查图片
file inputs/problem_image.jpg
identify inputs/problem_image.jpg  # ImageMagick
```

---

## 性能对比

### 速度对比 (秒/张)

```
RapidOCR: 1.3s  ████████████
HunYuan:  7.0s  ██████████████████████████████████████████████████████████████
Qwen3-VL: 5.0s  ████████████████████████████████████████████
```

### 准确度对比 (5张样本)

```
RapidOCR: 5/5   ██████████ 100%
HunYuan:  4/5   ████████░░  80%
Qwen3-VL: ?/?   ?
```

### 成本对比 (处理 1000 张发票)

```
RapidOCR:
  时间: 22 分钟
  成本: 免费
  总成本: ¥0

HunYuan (8GB GPU):
  时间: 117 分钟 (~2小时)
  成本: GPU 租赁 ($0.5/小时)
  总成本: ¥1-3

Qwen3-VL (4GB GPU):
  时间: 83 分钟
  成本: GPU 租赁 ($0.3/小时)
  总成本: ¥0.5-1.5
```

### 推荐使用场景

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 日常批量处理 | RapidOCR | 快、稳、免费 |
| 质量认证 | RapidOCR + HunYuan | 双重保障 |
| 研究对比 | 全部 | 理解各模型特性 |
| 离线处理 | RapidOCR | 无需网络和 GPU |
| 云端处理 | RapidOCR | 成本最低 |

---

## 文件说明

### 输入文件

- `inputs/*.jpg` - 待处理的发票图片

### 输出文件

#### RapidOCR 结果
- `ocr_results/rapidocr_results_[时间].json` - 完整结果
  ```json
  [{
    "filename": "invoice.jpg",
    "image_path": "inputs/invoice.jpg",
    "extracted_amount": 33.10,
    "raw_text": "...",
    "error": null
  }]
  ```

- `ocr_results/summary_[时间].txt` - 统计摘要
  ```
  ========================================
  RapidOCR Batch Processing Results
  ========================================

  Total invoices: 47
  Successfully processed: 47
  Total amount: ¥5,937.64
  ```

#### HunYuan 结果
- `ocr_results/hunyuan_results_[时间].json` - 完整结果
  ```json
  [{
    "filename": "invoice.jpg",
    "amount": 33.10,
    "text": "（普通发票）...",
    "error": null
  }]
  ```

#### Qwen3-VL 结果
- `ocr_results/qwen3vl_results_[时间].json` - 完整结果
  ```json
  [{
    "filename": "invoice.jpg",
    "amount": 33.10,
    "text": "...",
    "error": null
  }]
  ```

---

## 常见命令速查表

```bash
# 快速处理
uv run python rapidocr_batch_processor.py

# 对比模型
uv run python run_all_models_separate.py --models rapidocr hunyuan --limit 10

# 只测试 HunYuan
uv run python test_hunyuan_simple.py

# 评估准确度
uv run python evaluate_ocr_accuracy.py

# 查看最新结果
cat ocr_results/summary_*.txt | tail -20

# 清理 GPU
python3 -c "import torch; torch.cuda.empty_cache()"

# 查看 GPU 状态
nvidia-smi

# 统计发票
ls inputs/*.jpg | wc -l

# 对比结果
diff <(cat ocr_results/rapidocr_results_*.json | jq '.[].amount') \
     <(cat ocr_results/hunyuan_results_*.json | jq '.[].amount')
```

---

**最后更新**: 2025-12-01
**版本**: 1.0
**状态**: ✓ 完成
