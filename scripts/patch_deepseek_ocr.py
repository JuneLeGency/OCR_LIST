#!/usr/bin/env python3
"""
Patch DeepSeek-OCR-2 model code for transformers 5.x compatibility.

Target: ModelScope commit d9362d80 (d9362d8032017d261ef40fd7b18cd70db16cdbe7)
        https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR-2

Fixes applied (see DEEPSEEK_OCR2_TRANSFORMERS5_COMPAT.md for details):

Critical Bugs:
  1. inv_freq uninitialized after meta-device loading (non-persistent buffer)
     -> Always recompute inv_freq in _set_cos_sin_cache
  2. _init_weights uses .data.normal_() bypassing transformers 5.x guard
     -> Replace with nn.init.normal_() / nn.init.zeros_()

API Incompatibilities:
  3. LlamaAttention/LlamaFlashAttention2 import broken in 5.x
     -> Replace with standalone implementation matching 4.46.3 behavior
  4. rope_scaling auto-set to {'rope_type': 'default', ...} in 5.x
     -> Handle both 'type' and 'rope_type' keys, treat 'default' as None
  5. Cache.get_usable_length() removed in 5.x
     -> Replace with get_seq_length()
  6. cache_position support for generate() in 5.x
     -> Use cache_position[0].item() for past_key_values_length
  7. attention_mask shape mismatch in 5.x
     -> Adjust past_key_values_length when mask includes full sequence
  8. get_max_cache_shape() returns -1 (not None) for unlimited caches
     -> Add max_cache_length > 0 check

Usage:
    python scripts/patch_deepseek_ocr.py [--model-dir PATH]

Alternatively, apply the unified diff directly:
    cd ~/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR-2
    patch -p1 < /path/to/patches/deepseek_ocr2_transformers5.patch
"""

import argparse
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


DEFAULT_MODEL_PATH = Path.home() / ".cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR-2"

# ──────────────────────────────────────────────────────────────────────
# Standalone LlamaAttention + helpers (replaces transformers.models.llama import)
# ──────────────────────────────────────────────────────────────────────

LLAMA_ATTENTION_BLOCK = textwrap.dedent('''\


def _llama_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Standard (non-interleaved) rotary position embedding, matching transformers 4.46.3."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q1 = q[..., : q.shape[-1] // 2]
    q2 = q[..., q.shape[-1] // 2 :]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k1 = k[..., : k.shape[-1] // 2]
    k2 = k[..., k.shape[-1] // 2 :]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    q_embed = (q * cos) + (q_rotated * sin)
    k_embed = (k * cos) + (k_rotated * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Standalone LlamaAttention matching transformers 4.46.3 behavior.

    This avoids any 5.x LlamaAttention dependency while being compatible
    with the DeepseekV2DecoderLayer calling convention.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self._init_rope()

    def _init_rope(self):
        rope_scaling = self.config.rope_scaling
        # transformers 5.x auto-sets rope_scaling to {'rope_type': 'default', ...}
        scaling_type = None
        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if scaling_type == "default":
                scaling_type = None

        if scaling_type is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.config.rope_theta,
            )
        elif scaling_type == "yarn":
            rope_kwargs = {
                key: rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in rope_scaling
            }
            self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                scaling_factor=rope_scaling["factor"],
                base=self.config.rope_theta,
                **rope_kwargs,
            )
        else:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.config.rope_theta,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = _llama_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

''')


def patch_modeling_deepseekv2(model_path: Path) -> bool:
    """Patch modeling_deepseekv2.py for transformers 5.x compatibility."""
    file_path = model_path / "modeling_deepseekv2.py"
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return False

    content = file_path.read_text()
    original_content = content

    # ── Fix #3: Replace LlamaAttention import with standalone implementation ──
    # Handle both original and partially-patched forms
    llama_import_patterns = [
        # Original form
        r'from transformers\.models\.llama\.modeling_llama import \(\s*LlamaAttention,\s*LlamaFlashAttention2\s*\)',
        # Partially patched (LlamaFlashAttention2 already removed)
        r'from transformers\.models\.llama\.modeling_llama import \(\s*LlamaAttention,?\s*\)\n# Compatibility: LlamaFlashAttention2 removed in transformers 5\.x\n# Create a stub that uses LlamaAttention instead',
        r'from transformers\.models\.llama\.modeling_llama import \(\s*LlamaAttention,?\s*\)',
    ]
    replaced_llama = False
    for pattern in llama_import_patterns:
        new_content = re.sub(pattern, LLAMA_ATTENTION_BLOCK.rstrip(), content, count=1)
        if new_content != content:
            content = new_content
            replaced_llama = True
            break

    if not replaced_llama and 'class LlamaAttention(nn.Module):' not in content:
        print("  WARNING: Could not find LlamaAttention import pattern to replace")

    # ── Fix #1: inv_freq recomputation in _set_cos_sin_cache ──
    if 'Always recompute inv_freq' not in content:
        content = re.sub(
            r'(    def _set_cos_sin_cache\(self, seq_len, device, dtype\):\n'
            r'        self\.max_seq_len_cached = seq_len\n)'
            r'        t = torch\.arange\(\n'
            r'            self\.max_seq_len_cached, device=device, dtype=self\.inv_freq\.dtype\n'
            r'        \)\n'
            r'\n'
            r'        freqs = torch\.outer\(t, self\.inv_freq\.to\(t\.device\)\)',
            r'''\1        # Always recompute inv_freq here to handle meta-device model loading
        # (non-persistent buffers become uninitialized garbage after to_empty)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim)
        )
        self.inv_freq = inv_freq
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=inv_freq.dtype
        )

        freqs = torch.outer(t, inv_freq)''',
            content,
        )

    # ── Fix #2: _init_weights — use nn.init instead of .data.normal_() ──
    content = content.replace(
        'module.weight.data.normal_(mean=0.0, std=std)',
        'nn.init.normal_(module.weight, mean=0.0, std=std)',
    )
    content = content.replace(
        'module.bias.data.zero_()',
        'nn.init.zeros_(module.bias)',
    )

    # ── Fix #4: rope_scaling compat in DeepseekV2Attention.__init__ ──
    # softmax_scale block
    if 'scaling_factor = self.config.rope_scaling["factor"]' in content:
        content = content.replace(
            '''        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:''',
            '''        self.softmax_scale = self.q_head_dim ** (-0.5)
        rope_scaling = self.config.rope_scaling
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling.get("factor")
            if mscale_all_dim and scaling_factor:''',
        )

    # _init_rope in DeepseekV2Attention
    if '        if self.config.rope_scaling is None:\n' in content:
        content = re.sub(
            r'    def _init_rope\(self\):\n'
            r'        if self\.config\.rope_scaling is None:\n',
            '''    def _init_rope(self):
        rope_scaling = self.config.rope_scaling
        # transformers 5.x auto-sets rope_scaling to {'rope_type': 'default', ...}
        scaling_type = None
        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if scaling_type == "default":
                scaling_type = None

        if scaling_type is None:
''',
            content,
            count=1,
        )
        # Remove old commented-out code block
        content = re.sub(
            r'            # self\.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding\(\n'
            r'            #     self\.qk_rope_head_dim,\n'
            r'            #     max_position_embeddings=self\.max_position_embeddings,\n'
            r'            #     scaling_factor=scaling_factor,\n'
            r'            #     base=self\.rope_theta,\n'
            r'            # \)\n',
            '',
            content,
        )
        # Fix the else branch references
        content = content.replace(
            '            scaling_type = self.config.rope_scaling["type"]',
            '            scaling_factor = rope_scaling["factor"]',
        )
        content = content.replace(
            '            scaling_factor = self.config.rope_scaling["factor"]\n',
            '',
            1,
        )
        content = content.replace(
            'self.config.rope_scaling[key]',
            'rope_scaling[key]',
        )
        content = content.replace(
            'if key in self.config.rope_scaling',
            'if key in rope_scaling',
        )

    # ── Fix #5: get_usable_length -> get_seq_length ──
    content = re.sub(
        r'past_key_value\.get_usable_length\(kv_seq_len, self\.layer_idx\)',
        'past_key_value.get_seq_length(self.layer_idx)',
        content,
    )

    # ── Fix #6: cache_position support ──
    content = re.sub(
        r'            past_key_values_length = past_key_values\.get_usable_length\(seq_length\)',
        '''            # Compatibility: use cache_position if available (transformers 5.x)
            if cache_position is not None and len(cache_position) > 0:
                past_key_values_length = cache_position[0].item()
            else:
                past_key_values_length = past_key_values.get_seq_length()''',
        content,
    )

    # ── Fix #7: attention_mask shape compat ──
    if 'Compatibility: transformers 5.x may pass attention_mask with full sequence length' not in content:
        content = content.replace(
            '''        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(''',
            '''        else:
            # 4d mask is passed through the layers
            # Compatibility: transformers 5.x may pass attention_mask with full sequence length
            if attention_mask is not None and attention_mask.shape[-1] != seq_length + past_key_values_length:
                # attention_mask already includes past tokens, adjust past_key_values_length
                expected_kv_length = attention_mask.shape[-1]
                adjusted_past_length = expected_kv_length - seq_length
                if adjusted_past_length >= 0:
                    past_key_values_length = adjusted_past_length
            attention_mask = _prepare_4d_causal_attention_mask(''',
        )

    # ── Fix #8: max_cache_length > 0 check ──
    if 'max_cache_length > 0' not in content:
        content = content.replace(
            '''            if (
                    max_cache_length is not None
                    and attention_mask is not None''',
            '''            # Compatibility: get_max_cache_shape() returns -1 (not None) for unlimited caches
            if (
                    max_cache_length is not None
                    and max_cache_length > 0
                    and attention_mask is not None''',
        )

    # ── Fix from old script: is_torch_fx_available ──
    content = re.sub(
        r'from transformers\.utils\.import_utils import is_torch_fx_available',
        '''# Compatibility: is_torch_fx_available removed in transformers 5.x
def is_torch_fx_available():
    try:
        import torch.fx
        return True
    except ImportError:
        return False''',
        content,
    )

    # ── Fix from old script: seen_tokens / get_max_length ──
    content = re.sub(
        r'past_length = past_key_values\.seen_tokens',
        '# Compatibility: seen_tokens removed in transformers 5.x\n                past_length = cache_length',
        content,
    )
    content = re.sub(
        r'max_cache_length = past_key_values\.get_max_length\(\)',
        'max_cache_length = past_key_values.get_max_cache_shape()',
        content,
    )

    if content == original_content:
        print(f"  No changes needed (already patched): {file_path.name}")
        return False

    # Backup original
    backup_path = file_path.with_suffix('.py.bak')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  Backed up: {backup_path}")

    file_path.write_text(content)
    print(f"  Patched: {file_path.name}")
    return True


def patch_modeling_deepseekocr2(model_path: Path) -> bool:
    """Patch modeling_deepseekocr2.py for transformers 5.x compatibility."""
    file_path = model_path / "modeling_deepseekocr2.py"
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return False

    content = file_path.read_text()
    original_content = content

    # Fix seen_tokens and get_max_length
    content = re.sub(
        r'past_length = past_key_values\.seen_tokens',
        '# Compatibility: seen_tokens removed in transformers 5.x\n                past_length = cache_length',
        content,
    )
    content = re.sub(
        r'max_cache_length = past_key_values\.get_max_length\(\)',
        'max_cache_length = past_key_values.get_max_cache_shape()',
        content,
    )

    # Fix DeepseekOCR2Model.forward() - add cache_position parameter
    content = re.sub(
        r'''(class DeepseekOCR2Model\(DeepseekV2Model\):.*?def forward\(
        self,
        input_ids: torch\.LongTensor = None,
        attention_mask: Optional\[torch\.Tensor\] = None,
        position_ids: Optional\[torch\.LongTensor\] = None,
        past_key_values: Optional\[List\[torch\.FloatTensor\]\] = None,
        inputs_embeds: Optional\[torch\.FloatTensor\] = None,
        use_cache: Optional\[bool\] = None,
        output_attentions: Optional\[bool\] = None,
        output_hidden_states: Optional\[bool\] = None,
        images: Optional\[torch\.FloatTensor\] = None,
        images_seq_mask: Optional\[torch\.FloatTensor\] = None,
        images_spatial_crop: Optional\[torch\.FloatTensor\] = None,
        return_dict: Optional\[bool\] = None,)
    \) -> Union\[Tuple, BaseModelOutputWithPast\]:''',
        r'''\1
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:''',
        content,
        flags=re.DOTALL,
    )

    # Fix DeepseekOCR2Model.forward() - pass cache_position to super().forward()
    content = re.sub(
        r'''return super\(DeepseekOCR2Model, self\)\.forward\(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        \)''',
        '''return super(DeepseekOCR2Model, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids=position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position
        )''',
        content,
    )

    # Fix DeepseekOCR2ForCausalLM.forward() - add cache_position parameter
    content = re.sub(
        r'''(def forward\(
        self,
        input_ids: torch\.LongTensor = None,
        attention_mask: Optional\[torch\.Tensor\] = None,
        position_ids: Optional\[torch\.LongTensor\] = None,
        past_key_values: Optional\[List\[torch\.FloatTensor\]\] = None,
        inputs_embeds: Optional\[torch\.FloatTensor\] = None,
        labels: Optional\[torch\.LongTensor\] = None,
        use_cache: Optional\[bool\] = None,
        output_attentions: Optional\[bool\] = None,
        output_hidden_states: Optional\[bool\] = None,
        images: Optional\[torch\.FloatTensor\] = None,
        images_seq_mask: Optional\[torch\.FloatTensor\] = None,
        images_spatial_crop: Optional\[torch\.FloatTensor\] = None,
        return_dict: Optional\[bool\] = None,)

    \) -> Union\[Tuple, CausalLMOutputWithPast\]:''',
        r'''\1
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:''',
        content,
    )

    # Fix DeepseekOCR2ForCausalLM.forward() - pass cache_position to self.model()
    content = re.sub(
        r'''outputs  = self\.model\(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask = images_seq_mask,
            images_spatial_crop = images_spatial_crop,
            return_dict=return_dict

        \)''',
        '''outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=return_dict,
            cache_position=cache_position,
        )''',
        content,
    )

    # Fix prepare_inputs_for_generation - add cache_position to model_inputs
    content = re.sub(
        r'''model_inputs\.update\(
            \{
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs\.get\("use_cache"\),
                "attention_mask": attention_mask,
                "images": kwargs\.get\("images", None\),
                "images_seq_mask": kwargs\.get\("images_seq_mask", None\),
                "images_spatial_crop": kwargs\.get\("images_spatial_crop", None\),
            \}''',
        '''model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_seq_mask": kwargs.get("images_seq_mask", None),
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
            }''',
        content,
    )

    if content == original_content:
        print(f"  No changes needed (already patched): {file_path.name}")
        return False

    backup_path = file_path.with_suffix('.py.bak')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  Backed up: {backup_path}")

    file_path.write_text(content)
    print(f"  Patched: {file_path.name}")
    return True


def clear_hf_module_cache():
    """Clear HuggingFace transformers_modules cache for DeepSeek-OCR-2."""
    cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules"
    # The hyphen-to-underscore mapping creates this directory name
    patterns = ["DeepSeek_hyphen_OCR_hyphen_2", "DeepSeek-OCR-2", "deepseek-ai"]
    cleared = False
    for pattern in patterns:
        target = cache_dir / pattern
        if target.exists():
            shutil.rmtree(target)
            print(f"  Cleared HF cache: {target}")
            cleared = True
    if not cleared:
        print("  No HF module cache to clear")


def main():
    parser = argparse.ArgumentParser(description="Patch DeepSeek-OCR-2 for transformers 5.x")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--use-patch-file",
        action="store_true",
        help="Apply the unified .patch file instead of Python regex patching",
    )
    args = parser.parse_args()

    model_path = args.model_dir.expanduser()
    print(f"Patching DeepSeek-OCR-2 at: {model_path}")
    print("=" * 60)

    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        print("Please download the model first with:")
        print("  modelscope download --model deepseek-ai/DeepSeek-OCR-2")
        return 1

    if args.use_patch_file:
        patch_file = Path(__file__).parent.parent / "patches" / "deepseek_ocr2_transformers5.patch"
        if not patch_file.exists():
            print(f"Error: Patch file not found: {patch_file}")
            return 1
        # Backup
        src = model_path / "modeling_deepseekv2.py"
        bak = src.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"  Backed up: {bak}")
        result = subprocess.run(
            ["patch", "-p1", "--forward", "-i", str(patch_file)],
            cwd=model_path,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  patch command failed:\n{result.stderr}")
            return 1
    else:
        patched = False
        patched |= patch_modeling_deepseekv2(model_path)
        patched |= patch_modeling_deepseekocr2(model_path)

        if not patched:
            print("\nNo patches applied (already patched or no changes needed)")
            return 0

    print("\nClearing HuggingFace module cache...")
    clear_hf_module_cache()

    print("\nPatching complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
