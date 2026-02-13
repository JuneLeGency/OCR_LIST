"""Compatibility patches for hardware-specific issues.

Workaround for cuDNN Conv3D half-precision performance regression
=================================================================

cuDNN < 9.15 has a catastrophic performance bug in Conv3D with bf16/fp16 inputs:
the convolution planner selects an algorithm that is ~8000x slower than fp32.
This affects all VLM models using Conv3D patch embedding (Qwen3-VL, GLM-OCR,
Chandra-OCR, etc.).

Root cause: cuDNN algorithm selection bug (NOT a PyTorch or transformers issue).
- PyTorch issue: https://github.com/pytorch/pytorch/issues/168167
- PyTorch issue: https://github.com/pytorch/pytorch/issues/166643
- NVIDIA forum: https://forums.developer.nvidia.com/t/355210

Proper fix: upgrade cuDNN to 9.15+ (after `uv sync`, torch reverts it):
    pip install nvidia-cudnn-cu12>=9.15

Since torch 2.9.x pins cuDNN < 9.15, this module provides a runtime workaround
that casts Conv3D patch_embed layers to fp32 when the buggy cuDNN is detected.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Minimum cuDNN version with the Conv3D half-precision fix
_CUDNN_MIN_FIXED = 91500  # 9.15.00


def _need_conv3d_fp32_workaround() -> bool:
    """Check if Conv3D half-precision workaround is needed.

    Returns True when:
    - CUDA is available
    - cuDNN version < 9.15 (the buggy range)
    """
    if not torch.cuda.is_available():
        return False
    cudnn_ver = torch.backends.cudnn.version()
    if cudnn_ver is None:
        return False
    return cudnn_ver < _CUDNN_MIN_FIXED


def patch_conv3d_for_blackwell(model: nn.Module) -> None:
    """Patch Conv3D modules in VisionPatchEmbed to use fp32, working around
    the cuDNN half-precision Conv3D performance bug.

    On cuDNN < 9.15, Conv3D in bf16/fp16 is ~8000x slower than fp32 due to
    a bad algorithm selection in the cuDNN convolution planner. This affects
    RTX 5090 (Blackwell), H100/H200 (Hopper), and potentially other GPUs.

    This patch casts the Conv3D to fp32 and wraps forward to convert
    input/output dtype automatically. It is a no-op when cuDNN >= 9.15.

    To remove this workaround, upgrade cuDNN:
        pip install nvidia-cudnn-cu12>=9.15
    """
    if not _need_conv3d_fp32_workaround():
        return

    cudnn_ver = torch.backends.cudnn.version()
    patched = False

    for name, module in model.named_modules():
        if not name.endswith("patch_embed"):
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv3d):
                original_dtype = child.weight.dtype
                child.float()  # cast weights to fp32
                original_forward = child.forward

                def make_wrapper(orig_fwd, out_dtype):
                    def wrapper(x):
                        return orig_fwd(x.float()).to(out_dtype)
                    return wrapper

                child.forward = make_wrapper(original_forward, original_dtype)
                patched = True

    if patched:
        logger.warning(
            "Applied Conv3D fp32 workaround for cuDNN %d (< %d). "
            "To fix properly: pip install nvidia-cudnn-cu12>=9.15",
            cudnn_ver,
            _CUDNN_MIN_FIXED,
        )
