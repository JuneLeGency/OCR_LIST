# OCR Engine — 项目上下文

## 项目概述

统一 OCR 引擎，支持 10 个后端（7 GPU VLM + 3 传统/轻量）。
用于发票图片金额提取的 benchmark 对比测试。

## 已提交的改动

### 1. HunyuanOCR repetition_penalty 修复
- **文件**: `ocr_engine/backends/hunyuan_ocr.py:136`
- **改动**: `model.generate()` 添加 `repetition_penalty=1.1`
- **原因**: hunyuan-ocr 在高分辨率图片上进入重复生成循环，KV cache 膨胀导致 OOM

### 2. cuDNN Conv3D 半精度性能 workaround
- **文件**: `ocr_engine/compat.py` (新增), 3 个 backend 文件引用
- **问题**: cuDNN < 9.15 的 Conv3D 在 bf16/fp16 下有 ~8000x 性能退化
  - 影响所有使用 Qwen3-VL 架构 Conv3D patch_embed 的模型
  - 根因: cuDNN 卷积规划器选择了错误的算法 (PyTorch#168167)
- **修复**: 检测到 cuDNN < 9.15 时，将 patch_embed 的 Conv3D 转为 fp32 运算
- **效果**: qwen3-vl 10.4s→2.4s (4.3x), glm-ocr 11.9s→1.6s (7.4x), chandra-ocr 22.7s→12.5s (1.8x)
- **正式修复**: `pip install nvidia-cudnn-cu12>=9.15` (torch 2.9.x 依赖冲突，需手动安装)

### 3. pyproject.toml 更新
- 新增 `[dependency-groups]` all-backends 组 + `default-groups` 配置
- `uv sync` 自动安装所有后端依赖，不再需要手动 `uv pip install -e ".[all]"`
- 保留 `[project.optional-dependencies]` 用于 pip 按需安装（如 `pip install ocr-engine[rapidocr]`）

### 4. README.md 更新
- 模型名称添加超链接
- 新增 Model References 表格
- 更新 dots-ocr 参数（1.7B → 3.0B）和 benchmark 数据

## 速度测试结果 (Feb 13, 2026)

测试图片: `inputs/25337000000272477597.jpg` (¥32.14), GPU: RTX 5090

| 模型 | 加载 | 推理 | VRAM 峰值 | 金额 | 正确 |
|------|------|------|-----------|------|------|
| tesseract | 0.3s | 0.7s | 1,569 MiB | ¥0.96 | ✗ |
| rapidocr | 0.5s | 1.2s | 1,569 MiB | ¥32.14 | ✓ |
| glm-ocr | 0.7s | 1.7s | 3,960 MiB | ¥32.14 | ✓ |
| qwen3-vl | 3.0s | 2.4s | 5,813 MiB | ¥32.14 | ✓ |
| hunyuan-ocr | 19.2s | 2.5s | 5,836 MiB | ¥32.14 | ✓ |
| deepseek-ocr | 1.9s | 2.8s | 10,248 MiB | ¥32.14 | ✓ |
| dots-ocr | 2.8s | 7.6s | 15,609 MiB | ¥32.14 | ✓ |
| chandra-ocr | 22.0s | 12.5s | 18,752 MiB | ¥32.14 | ✓ |
| lighton-ocr | 0.5s | 3.3s | ~2,100 MiB | ¥32.14 | ✓ |
| firered-ocr | 0.6s | 5.1s | ~4,100 MiB | N/A | ✗* |

## Benchmark 进展 (Mar 4, 2026)

### Benchmark 准确率 (47 invoices, majority-vote consensus)
| 模型 | 准确率 | Match | Mismatch | Error |
|------|--------|-------|----------|-------|
| glm-ocr | **100.0%** | 47/47 | 0 | 0 |
| qwen3-vl | **100.0%** | 47/47 | 0 | 0 |
| hunyuan-ocr | **100.0%** | 44/47 | 0 | 3 |
| lighton-ocr | **100.0%** | 32/47 | 0 | 15 |
| dots-ocr | **100.0%** | 18/47 | 0 | 29 |
| firered-ocr | 97.5% | 39/47 | 1 | 7 |
| rapidocr | 97.7% | 43/47 | 1 | 3 |
| deepseek-ocr | 79.2% | 19/47 | 5 | 23 |
| tesseract | 65.8% | 27/47 | 14 | 6 |

### 新增模型观察
- **lighton-ocr**: 100% 准确率（无 mismatch），15 个 error 是高分辨率发票 max_new_tokens 耗尽导致金额区域被截断。中位推理 ~8s/img
- **firered-ocr**: 97.5% 准确率，1 个 mismatch（¥14.8 vs consensus ¥14.37），7 个 error 是 HTML 输出含大量 `&nbsp;` 填充消耗 token，金额字段未生成。中位推理 ~11s/img
- **firered-ocr** 是 Qwen3-VL-2B 微调，默认 prompt 面向文档→Markdown 转换，不适合金额提取场景
- **lighton-ocr** 的 Pixtral ViT 编码器不受 cuDNN Conv3D bug 影响

## 性能分析结论

### chandra-ocr 12s 推理时间（正常）
- chandra-ocr 是 Qwen3-VL-7B 微调版，**8.77B 参数**（qwen3-vl 的 4 倍）
- Conv3D 补丁已生效：prefill 仅 127ms (1.1%)，瓶颈在 decode 阶段 (98.9%)
- Decode 速度 62.6 tok/s，生成 723 tokens → 11.6s，**符合理论预期**
- bf16 模型大小 17.5 GB，有效带宽利用 1098/1792 GB/s = 61%（所有模型中最高）
- 加速方案: vLLM 推理 / 量化 (AWQ/GPTQ 4-bit) / 减少输出 token

### 各模型 decode 速度对比 (RTX 5090, 1792 GB/s)
| 模型 | 参数量 | Decode | 带宽利用率 |
|------|--------|--------|-----------|
| glm-ocr | 1.11B | 160 tok/s | 20% |
| qwen3-vl | 2.13B | 109 tok/s | 26% |
| chandra-ocr | 8.77B | 62.6 tok/s | 61% |

## 已知问题

### cuDNN Conv3D 半精度 bug（已 workaround）
- **影响**: PyTorch 2.9.x + cuDNN < 9.15，所有 GPU 架构
- **症状**: Conv3D bf16/fp16 比 fp32 慢 ~8000 倍，同时内存膨胀 ~3x
- **受影响模型**: qwen3-vl, glm-ocr, chandra-ocr, firered-ocr（均使用 Qwen3-VL 架构的 Conv3D patch_embed）
- **当前 workaround**: `ocr_engine/compat.py` 自动检测并将 Conv3D 转为 fp32
- **正式修复**: `pip install nvidia-cudnn-cu12>=9.15`（torch 2.9.x 依赖会被 `uv sync` 覆盖回旧版）
- **参考**:
  - https://github.com/pytorch/pytorch/issues/168167
  - https://github.com/pytorch/pytorch/issues/166643
  - https://forums.developer.nvidia.com/t/355210

## 技术栈

- Python 3.12, uv 包管理
- PyTorch 2.9, transformers 5.x (custom fork for HunyuanOCR)
- GPU: RTX 5090 (32 GB), cuDNN 9.10 (with fp32 workaround)
- 模型缓存: `~/.cache/modelscope/hub/models/`
