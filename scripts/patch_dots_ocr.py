#!/usr/bin/env python3
"""
Patch dots.ocr model code for transformers 5.x compatibility + ModelScope support.

Target: ModelScope cache ~/.cache/modelscope/hub/models/rednote-hilab/dots.ocr/

Fixes applied:

Critical Bugs:
  1. transformers 5.x meta device init corrupts vision tower weights (214/304 params)
     and VisionRotaryEmbedding.inv_freq (non-persistent buffer → garbage values).
     -> Add custom from_pretrained to DotsOCRForCausalLM that creates model on CPU
        via _from_config, then loads safetensors manually.
        Same pattern as HunyuanVL fix in transformers fork.
  2. from_config + load_state_dict doesn't load generation_config.json, so model
     lacks EOS token (151673), causing generation to run to max_new_tokens.
     -> Custom from_pretrained loads GenerationConfig explicitly.
  3. DotsVLProcessor.__init__ drops **kwargs, breaking transformers 5.x
     ProcessorMixin which passes extra kwargs.
     -> Pass **kwargs to super().__init__().

Usage:
    python scripts/patch_dots_ocr.py [--model-dir PATH]
"""

import argparse
import shutil
import sys
import textwrap
from pathlib import Path


DEFAULT_MODEL_PATH = Path.home() / ".cache/modelscope/hub/models/rednote-hilab/dots.ocr"

# Marker to detect already-patched files
PATCH_MARKER = "# [patch_dots_ocr] custom from_pretrained for transformers 5.x"

# ──────────────────────────────────────────────────────────────────────
# Patch A: Custom from_pretrained for DotsOCRForCausalLM
# ──────────────────────────────────────────────────────────────────────

# Note: 4-space indent = class level, 8-space = method body
FROM_PRETRAINED_BLOCK = '''
    {marker}
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Custom from_pretrained bypassing transformers 5.x meta device init.

        transformers 5.x meta device initialization corrupts:
          1. Vision tower weights (214/304 params get wrong values)
          2. VisionRotaryEmbedding.inv_freq (non-persistent buffer -> garbage)

        Adaptive strategy based on available VRAM:
          - Enough VRAM (>= 2.5x model): init on CUDA directly (fastest)
          - Tight VRAM: init on meta device, load to CUDA, fix inv_freq
            (peak = 1x model instead of 2x)
        """
        import os
        from safetensors.torch import load_file
        from transformers import AutoConfig, GenerationConfig

        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = kwargs.pop("dtype", None)
        device_map = kwargs.pop("device_map", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        _ = kwargs.pop("attn_implementation", None)
        effective_dtype = dtype or torch_dtype

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code, **kwargs
        )

        use_cuda = device_map is not None and torch.cuda.is_available()
        use_meta = False
        if use_cuda:
            model_bytes = sum(
                os.path.getsize(os.path.join(pretrained_model_name_or_path, f))
                for f in os.listdir(pretrained_model_name_or_path)
                if f.endswith(".safetensors")
            )
            free_vram = torch.cuda.mem_get_info()[0]
            # 2x needed: _from_config alloc + state_dict; use 2.5x as safety margin
            use_meta = free_vram < model_bytes * 2.5

        if use_meta:
            # Tight VRAM: meta init (0 bytes) -> load to CUDA -> fix inv_freq
            with torch.device("meta"):
                model = cls._from_config(config, torch_dtype=effective_dtype)
            load_device = "cuda"
        elif use_cuda:
            # Plenty of VRAM: CUDA init (fastest, ~0.1s vs 34s on CPU)
            with torch.device("cuda"):
                model = cls._from_config(config, torch_dtype=effective_dtype)
            load_device = "cuda"
        else:
            with torch.device("cpu"):
                model = cls._from_config(config, torch_dtype=effective_dtype)
            load_device = "cpu"

        state_dict = {{}}
        for f in sorted(os.listdir(pretrained_model_name_or_path)):
            if f.endswith(".safetensors"):
                state_dict.update(load_file(
                    os.path.join(pretrained_model_name_or_path, f),
                    device=load_device,
                ))
        model.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict

        if use_meta:
            # Non-persistent buffers (inv_freq) are still on meta after assign.
            # Recompute them on the target device.
            for module in model.modules():
                if hasattr(module, "inv_freq") and module.inv_freq.is_meta:
                    dim = module.inv_freq.numel() * 2
                    module.inv_freq = 1.0 / (10000.0 ** (
                        torch.arange(0, dim, 2, dtype=torch.float, device=load_device) / dim
                    ))

        model.tie_weights()

        try:
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path
            )
        except OSError:
            pass

        model.eval()
        return model

'''.format(marker=PATCH_MARKER)


def patch_modeling_dots_ocr(model_path: Path) -> bool:
    """Patch modeling_dots_ocr.py: add custom from_pretrained to DotsOCRForCausalLM."""
    file_path = model_path / "modeling_dots_ocr.py"
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return False

    content = file_path.read_text()

    if PATCH_MARKER in content:
        print(f"  Already patched: {file_path.name}")
        return False

    # Insert from_pretrained after `config_class = DotsOCRConfig`
    target = "    config_class = DotsOCRConfig"
    if target not in content:
        print(f"  WARNING: Could not find '{target}' in {file_path.name}")
        return False

    new_content = content.replace(
        target,
        target + FROM_PRETRAINED_BLOCK,
    )

    if new_content == content:
        print(f"  No changes made to {file_path.name}")
        return False

    # Backup original
    backup_path = file_path.with_suffix('.py.bak')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  Backed up: {backup_path}")

    file_path.write_text(new_content)
    print(f"  Patched: {file_path.name} (added custom from_pretrained)")
    return True


def patch_configuration_dots(model_path: Path) -> bool:
    """Patch configuration_dots.py: pass **kwargs in DotsVLProcessor.__init__."""
    file_path = model_path / "configuration_dots.py"
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return False

    content = file_path.read_text()
    original_content = content

    old = "super().__init__(image_processor, tokenizer, chat_template=chat_template)"
    new = "super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)"

    if old not in content:
        if new in content:
            print(f"  Already patched: {file_path.name}")
            return False
        print(f"  WARNING: Could not find super().__init__ pattern in {file_path.name}")
        return False

    content = content.replace(old, new)

    if content == original_content:
        return False

    backup_path = file_path.with_suffix('.py.bak')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"  Backed up: {backup_path}")

    file_path.write_text(content)
    print(f"  Patched: {file_path.name} (added **kwargs to super().__init__)")
    return True


def clear_hf_module_cache():
    """Clear HuggingFace transformers_modules cache for dots.ocr.

    Forces transformers to re-import from the ModelScope path instead of
    using stale cached modules.
    """
    cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules"
    if not cache_dir.exists():
        print("  No HF module cache to clear")
        return

    # HuggingFace encodes special chars: / → _hyphen_, . → _dot_
    patterns = [
        "rednote_hyphen_hilab",
        "rednote-hilab",
        "dots_dot_ocr",
    ]
    cleared = False
    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        # Match known patterns
        if any(p in name for p in patterns):
            shutil.rmtree(entry)
            print(f"  Cleared HF cache: {entry}")
            cleared = True
            continue
        # Also clear commit-hash dirs that contain dots.ocr code
        if len(name) == 40 and (entry / "modeling_dots_ocr.py").exists():
            shutil.rmtree(entry)
            print(f"  Cleared HF cache (commit hash): {entry}")
            cleared = True

    if not cleared:
        print("  No HF module cache to clear")


def main():
    parser = argparse.ArgumentParser(
        description="Patch dots.ocr for transformers 5.x compatibility"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model directory (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    model_path = args.model_dir.expanduser()
    print(f"Patching dots.ocr at: {model_path}")
    print("=" * 60)

    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        print("Please download the model first with:")
        print("  python setup_modelscope_models.py --download dots-ocr")
        return 1

    patched = False
    patched |= patch_modeling_dots_ocr(model_path)
    patched |= patch_configuration_dots(model_path)

    if not patched:
        print("\nNo patches applied (already patched or no changes needed)")
    else:
        print(f"\nPatched {model_path}")

    print("\nClearing HuggingFace module cache...")
    clear_hf_module_cache()

    print("\nPatching complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
