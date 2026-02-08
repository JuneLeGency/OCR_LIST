#!/usr/bin/env python3
"""
统一下载和管理 OCR 模型 (使用 ModelScope)
"""

import os
import shutil
from pathlib import Path

# ModelScope 模型配置
MODELS = {
    "glm-ocr": {
        "modelscope_id": "ZhipuAI/GLM-OCR",
        "huggingface_id": "zai-org/GLM-OCR",
        "description": "智谱 GLM-OCR 专用模型",
    },
    "qwen3-vl": {
        "modelscope_id": "Qwen/Qwen3-VL-2B-Instruct",
        "huggingface_id": "Qwen/Qwen3-VL-2B-Instruct",
        "description": "阿里 Qwen3-VL 视觉语言模型",
    },
    "hunyuan-ocr": {
        "modelscope_id": "Tencent-Hunyuan/HunyuanOCR",
        "huggingface_id": "tencent/HunyuanOCR",
        "description": "腾讯 HunyuanOCR 端到端OCR专家模型",
    },
    "deepseek-ocr2": {
        "modelscope_id": "deepseek-ai/DeepSeek-OCR-2",
        "huggingface_id": "deepseek-ai/DeepSeek-OCR-2",
        "description": "DeepSeek OCR-2 视觉因果流模型",
    },
    "dots-ocr": {
        "modelscope_id": "rednote-hilab/dots.ocr",
        "huggingface_id": "rednote-hilab/dots.ocr",
        "description": "RedNote dots.ocr 多语言文档版面解析模型",
    },
}

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MS_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / "models"


def get_hf_cache_path(model_id: str) -> Path:
    """获取 HuggingFace 缓存路径"""
    return HF_CACHE / f"models--{model_id.replace('/', '--')}"


def get_ms_cache_path(model_id: str) -> Path:
    """获取 ModelScope 缓存路径"""
    return MS_CACHE / model_id.replace("/", "/")


def check_model_status():
    """检查所有模型的缓存状态"""
    print("=" * 70)
    print("模型缓存状态检查")
    print("=" * 70)

    for name, config in MODELS.items():
        ms_path = get_ms_cache_path(config["modelscope_id"])
        hf_path = get_hf_cache_path(config["huggingface_id"])

        ms_exists = ms_path.exists()
        hf_exists = hf_path.exists()

        ms_size = get_dir_size(ms_path) if ms_exists else 0
        hf_size = get_dir_size(hf_path) if hf_exists else 0

        print(f"\n{name}:")
        print(f"  描述: {config['description']}")
        print(f"  ModelScope ({config['modelscope_id']}): ", end="")
        if ms_exists:
            print(f"✅ {format_size(ms_size)}")
        else:
            print("❌ 未下载")

        print(f"  HuggingFace ({config['huggingface_id']}): ", end="")
        if hf_exists:
            print(f"⚠️  {format_size(hf_size)} (可清理)")
        else:
            print("- 无缓存")


def get_dir_size(path: Path) -> int:
    """获取目录大小"""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def cleanup_huggingface_cache(dry_run: bool = True):
    """清理 HuggingFace 缓存"""
    print("\n" + "=" * 70)
    print("清理 HuggingFace 缓存" + (" (预览模式)" if dry_run else ""))
    print("=" * 70)

    # 要删除的路径
    to_delete = [
        HF_CACHE / "models--tencent--HunYuanOCR",  # 重复的大小写版本
        HF_CACHE / "models--zai-org--GLM-OCR",  # 不完整
        HF_CACHE / "models--datalab-to--chandra",  # 不相关
    ]

    total_freed = 0
    for path in to_delete:
        if path.exists():
            size = get_dir_size(path)
            total_freed += size
            print(f"  删除: {path.name} ({format_size(size)})")
            if not dry_run:
                shutil.rmtree(path)

    print(f"\n总计释放: {format_size(total_freed)}")

    if dry_run:
        print("\n使用 --cleanup 参数执行实际删除")


def download_models(models_to_download: list = None):
    """从 ModelScope 下载模型"""
    from modelscope import snapshot_download

    if models_to_download is None:
        models_to_download = list(MODELS.keys())

    print("\n" + "=" * 70)
    print("从 ModelScope 下载模型")
    print("=" * 70)

    for name in models_to_download:
        if name not in MODELS:
            print(f"未知模型: {name}")
            continue

        config = MODELS[name]
        ms_path = get_ms_cache_path(config["modelscope_id"])

        if ms_path.exists():
            size = get_dir_size(ms_path)
            print(f"\n{name}: 已存在 ({format_size(size)}), 跳过")
            continue

        print(f"\n下载 {name}: {config['modelscope_id']}")
        print(f"  {config['description']}")

        try:
            snapshot_download(config["modelscope_id"])
            print(f"  ✅ 下载完成")
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")


def cleanup_after_download():
    """下载完成后清理对应的 HuggingFace 缓存"""
    print("\n" + "=" * 70)
    print("清理已迁移到 ModelScope 的 HuggingFace 缓存")
    print("=" * 70)

    for name, config in MODELS.items():
        ms_path = get_ms_cache_path(config["modelscope_id"])
        hf_path = get_hf_cache_path(config["huggingface_id"])

        if ms_path.exists() and hf_path.exists():
            size = get_dir_size(hf_path)
            print(f"  删除: {hf_path.name} ({format_size(size)})")
            shutil.rmtree(hf_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OCR 模型缓存管理")
    parser.add_argument("--status", action="store_true", help="检查模型状态")
    parser.add_argument("--cleanup", action="store_true", help="清理无用的 HuggingFace 缓存")
    parser.add_argument("--download", nargs="*", help="下载模型 (不指定则下载全部)")
    parser.add_argument("--cleanup-migrated", action="store_true", help="清理已迁移的模型")

    args = parser.parse_args()

    if args.status or not any([args.cleanup, args.download is not None, args.cleanup_migrated]):
        check_model_status()

    if args.cleanup:
        cleanup_huggingface_cache(dry_run=False)
    elif not args.status and args.download is None and not args.cleanup_migrated:
        cleanup_huggingface_cache(dry_run=True)

    if args.download is not None:
        models = args.download if args.download else None
        download_models(models)

    if args.cleanup_migrated:
        cleanup_after_download()


if __name__ == "__main__":
    main()
