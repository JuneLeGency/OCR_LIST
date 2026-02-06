# DeepSeek-OCR-2: Transformers 5.x 兼容性修复指南

## 概述

DeepSeek-OCR-2 模型（从 ModelScope 下载）的 `modeling_deepseekv2.py` 基于 transformers 4.46.3 编写。在 transformers 5.x 下加载时会产出乱码/垃圾输出，原因是两个**致命 Bug** 和若干 API 不兼容。

本文档记录了所有修改，供从 ModelScope 下载模型后需要在 transformers 5.x 环境下使用的开发者参考。

- ModelScope 模型页: https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR-2
- 对应 commit: `d9362d80` (full: `d9362d8032017d261ef40fd7b18cd70db16cdbe7`)
- 模型路径（ModelScope 缓存）: `~/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR-2/`
- 需要修改的文件: `modeling_deepseekv2.py`（主要）、`modeling_deepseekocr2.py`（cache_position 传递等）
- 测试环境: transformers 5.0.5.dev0（自定义 fork），PyTorch 2.x，CUDA
- 工程内工具:
  - `scripts/patch_deepseek_ocr.py` — Python 脚本，自动应用所有修复（推荐）
  - `patches/deepseek_ocr2_transformers5.patch` — unified diff，可直接 `patch -p1` 应用

---

## Bug #1: Rotary Embedding inv_freq 未初始化（meta device + non-persistent buffer）

### 现象

模型加载后，所有注意力层的 rotary embedding 的 `inv_freq` 为 CUDA 未初始化内存中的垃圾值（如 `[2.39e-14, 0.0, 0.0, 7.0e-45]`），导致：
- `cos` 全部为 1.0，`sin` 全部为 0.0
- 模型没有位置编码，输出乱码

### 根因分析

1. transformers 5.x 的 `from_pretrained` 先在 **meta device** 上创建模型骨架
2. 调用 `to_empty(device='cuda')` 将张量实体化到 CUDA，此时所有内存**未初始化**
3. 从 checkpoint 加载权重（safetensors），覆盖模型参数
4. 但 `inv_freq` 通过 `register_buffer("inv_freq", inv_freq, persistent=False)` 注册为**非持久化 buffer**
5. 非持久化 buffer **不在 checkpoint 中保存**，因此加载后仍为未初始化的 CUDA 垃圾值

### 修复方案

在 `DeepseekV2RotaryEmbedding._set_cos_sin_cache` 中**始终重新计算 `inv_freq`**，而不是依赖 `__init__` 中注册的 buffer：

```python
def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    # Always recompute inv_freq here to handle meta-device model loading
    # (non-persistent buffers become uninitialized garbage after to_empty)
    inv_freq = 1.0 / (
        self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim)
    )
    self.inv_freq = inv_freq
    t = torch.arange(
        self.max_seq_len_cached, device=device, dtype=inv_freq.dtype
    )
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
```

**为什么不用阈值检测？** 尝试过 `inv_freq.abs().max() == 0` 和 `< 1e-6`，均失败——未初始化的 CUDA 内存可能包含极大值（如 `3.05e+32`）或极小值，没有可靠的阈值。直接重算是最可靠的方案，因为所需参数（`base`, `dim`）在类属性中始终可用。

---

## Bug #2: 权重被 `_init_weights` 重新随机初始化（.data.normal_() 绕过 5.x guard）

### 现象

模型加载后，所有 decoder attention 的权重被替换为随机值。验证方法：

```python
from safetensors.torch import load_file
state = load_file("model.safetensors")
model_weight = model.model.layers[0].self_attn.q_proj.weight.cpu().float()
file_weight = state['model.layers.0.self_attn.q_proj.weight'].float()
diff = (model_weight - file_weight).abs().max().item()
# 结果: diff = 0.78 (应该为 0.0)
```

### 根因分析

1. transformers 5.x 加载流程: `from_pretrained` → 加载 checkpoint 权重 → `_finalize_model_loading` → `_initialize_missing_keys` → `initialize_weights()` → 调用模型的 `_init_weights`
2. `initialize_weights()` 被 `@init.guard_torch_init_functions()` 装饰器保护
3. 该装饰器 **patch** 了 `torch.nn.init.*` 系列函数（如 `nn.init.normal_`），在调用前检查张量的 `_is_hf_initialized` 标志——已从 checkpoint 加载的张量有此标志，因此 `nn.init.normal_` 会**跳过**它们
4. **但是**，原始代码使用 `module.weight.data.normal_(mean=0.0, std=std)` ——这是**直接的 Tensor 方法调用**，完全绕过了 guard 机制
5. 结果：已加载的权重被随机值覆盖

### 修复方案

将 `_init_weights` 中的直接张量操作改为 `nn.init.*` 函数调用：

```python
# 原始代码（有 Bug）:
def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)      # 绕过 guard!
        if module.bias is not None:
            module.bias.data.zero_()                         # 绕过 guard!
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)      # 绕过 guard!

# 修复后:
def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)   # 受 guard 保护
        if module.bias is not None:
            nn.init.zeros_(module.bias)                      # 受 guard 保护
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)   # 受 guard 保护
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
```

---

## 其他兼容性修改

### 1. 替换 LlamaAttention 为独立实现

**原因**: DeepSeek-OCR-2 使用 `use_mla=False`（MHA 路径），依赖 `transformers.models.llama.modeling_llama.LlamaAttention`。但 5.x 版本的 LlamaAttention 签名发生了重大变化：
- `forward()` 使用 `position_embeddings` 参数替代 `position_ids`
- 返回 2-tuple 而非 3-tuple
- 不再内置 `rotary_emb`
- `LlamaFlashAttention2` 已移除

**方案**: 用一个独立的 `LlamaAttention` 类替换 import，复刻 4.46.3 的完整行为（约 120 行）。同时 `LlamaFlashAttention2 = LlamaAttention`。

### 2. rope_scaling 兼容

transformers 5.x 会自动将 `rope_scaling=None` 设置为 `{'rope_type': 'default', 'rope_theta': 10000.0}`。原始代码直接 `rope_scaling["type"]` 和 `rope_scaling["factor"]` 会 KeyError。

修复: 检查 `"type"` 和 `"rope_type"` 两个 key，将 `"default"` 视为无 scaling；用 `.get("factor")` 替代 `["factor"]`。

### 3. get_usable_length → get_seq_length

`Cache.get_usable_length()` 在 5.x 中已移除，替换为 `get_seq_length()`。涉及 3 处调用。

### 4. cache_position 支持

transformers 5.x 的 `generate()` 传入 `cache_position` 张量。在 `DeepseekV2Model.forward` 中需要优先使用 `cache_position[0].item()` 计算 `past_key_values_length`。

### 5. attention_mask 形状兼容

5.x 可能传入包含完整序列长度的 attention_mask，需要在调用 `_prepare_4d_causal_attention_mask` 之前调整 `past_key_values_length`。

### 6. max_cache_length 兼容

5.x 的 `get_max_cache_shape()` 对无限缓存返回 `-1`（而非 `None`），需要额外检查 `max_cache_length > 0`。

### 7. is_torch_fx_available 移除

`from transformers.utils.import_utils import is_torch_fx_available` 在 5.x 中不存在。替换为本地 fallback 函数。

### 8. seen_tokens / get_max_length 移除

- `past_key_values.seen_tokens` 属性已移除，改用 `cache_length`（局部变量，已在上文计算）。
- `past_key_values.get_max_length()` 已移除，改用 `past_key_values.get_max_cache_shape()`。

### 9. modeling_deepseekocr2.py 兼容性修改

`modeling_deepseekocr2.py` 继承 `DeepseekV2Model`，需要额外修改：
- `DeepseekOCR2Model.forward()` 和 `DeepseekOCR2ForCausalLM.forward()` 添加 `cache_position` 参数并向下传递
- `prepare_inputs_for_generation()` 将 `cache_position` 加入 `model_inputs`
- 同样修复 `seen_tokens` 和 `get_max_length` 调用

---

## 操作指南：从 ModelScope 下载后如何修改

### 前提条件

- 已通过 ModelScope 下载 DeepSeek-OCR-2 模型
- 使用 transformers >= 5.0

### 方式 A: Python 脚本（推荐）

脚本自动处理备份、补丁、缓存清理，支持幂等执行（已补丁过的文件不会重复修改）。

```bash
# 默认路径
python3 scripts/patch_deepseek_ocr.py

# 自定义模型路径
python3 scripts/patch_deepseek_ocr.py --model-dir /path/to/DeepSeek-OCR-2

# 也可以用 --use-patch-file 走 patch 命令而非 regex
python3 scripts/patch_deepseek_ocr.py --use-patch-file
```

### 方式 B: 手动应用 patch 文件

```bash
MODEL_DIR=~/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR-2

# 备份
cp $MODEL_DIR/modeling_deepseekv2.py $MODEL_DIR/modeling_deepseekv2.py.bak

# 应用 patch（注意 -p1，因为 patch 使用 a/b/ 前缀）
cd $MODEL_DIR
patch -p1 < /path/to/patches/deepseek_ocr2_transformers5.patch

# 清除 HuggingFace 模块缓存（重要！否则 transformers 会使用旧的缓存版本）
rm -rf ~/.cache/huggingface/modules/transformers_modules/DeepSeek_hyphen_OCR_hyphen_2/
```

> **注意**: patch 文件仅覆盖 `modeling_deepseekv2.py`。`modeling_deepseekocr2.py` 的修改需要通过 Python 脚本（方式 A）应用。

### 验证

```python
import torch, tempfile
from transformers import AutoModel, AutoTokenizer

model_path = "~/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16,
).eval().cuda()

# 验证权重正确加载
from safetensors.torch import load_file
import glob, os
sf = glob.glob(os.path.join(model_path, "*.safetensors"))[0]
state = load_file(sf)
diff = (model.model.layers[0].self_attn.q_proj.weight.cpu().float()
        - state['model.layers.0.self_attn.q_proj.weight'].float()).abs().max().item()
print(f"Weight diff: {diff:.2e}")  # 应为 0.00e+00
del state

# 验证 rotary embedding
rope = model.model.layers[0].self_attn.rotary_emb
print(f"inv_freq[:3]: {rope.inv_freq[:3].tolist()}")  # 应为 [1.0, 0.87..., 0.76...]

# 运行 OCR
with tempfile.TemporaryDirectory() as tmp:
    result = model.infer(
        tokenizer,
        prompt="<image>\nFree OCR. ",
        image_file="your_image.jpg",
        output_path=tmp, base_size=1024, image_size=768,
        crop_mode=True, save_results=False, eval_mode=True,
    )
print(result[:200])
```

---

## 完整 Patch

完整的 unified diff 保存在 `patches/deepseek_ocr2_transformers5.patch`（312 行），此处不再内嵌。

查看 patch 内容：

```bash
cat patches/deepseek_ocr2_transformers5.patch
```

---

## 修改摘要

### modeling_deepseekv2.py

| 修改 | 文件位置 | 类型 | 影响 |
|------|---------|------|------|
| Bug #1: inv_freq 重算 | `_set_cos_sin_cache` | 致命 Bug | 无位置编码 → 乱码 |
| Bug #2: nn.init 替换 | `_init_weights` | 致命 Bug | 权重随机化 → 乱码 |
| 独立 LlamaAttention | 文件顶部 | API 不兼容 | import 失败/行为不一致 |
| rope_scaling 兼容 | `_init_rope` (2处) | API 不兼容 | KeyError 崩溃 |
| get_seq_length | 3处 | API 移除 | AttributeError 崩溃 |
| cache_position | `forward` | API 新增 | generate() 失败 |
| attention_mask 形状 | `forward` | 行为变更 | 注意力计算错误 |
| max_cache_length | `prepare_inputs` | 返回值变更 | 生成时截断错误 |
| is_torch_fx_available | import 区域 | API 移除 | ImportError 崩溃 |
| seen_tokens | `prepare_inputs` | 属性移除 | AttributeError 崩溃 |
| get_max_length | `prepare_inputs` | API 移除 | AttributeError 崩溃 |

### modeling_deepseekocr2.py

| 修改 | 文件位置 | 类型 | 影响 |
|------|---------|------|------|
| cache_position 传递 | `forward` (2处) + `prepare_inputs` | API 新增 | generate() 失败 |
| seen_tokens | `prepare_inputs` | 属性移除 | AttributeError 崩溃 |
| get_max_length | `prepare_inputs` | API 移除 | AttributeError 崩溃 |
