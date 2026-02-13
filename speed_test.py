#!/usr/bin/env python3
"""单张图片速度测试 — 逐模型测试，测试间清理显存。"""

import gc
import time
import subprocess
from pathlib import Path

import torch
from ocr_engine import OCREngine
from ocr_engine.config import SUPPORTED_MODELS


TEST_IMAGE = Path("inputs/25337000000272477597.jpg")  # ground truth: ¥32.14
EXPECTED_AMOUNT = 32.14
ALL_MODELS = list(SUPPORTED_MODELS.keys())


def get_gpu_mem_mb() -> tuple[float, float]:
    """返回 (已用MB, 总MB)。"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    return used, reserved


def get_nvidia_smi_mb() -> float:
    """通过 nvidia-smi 获取真实物理显存占用 (MiB)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def force_cleanup():
    """强制清理 GPU 显存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def test_single_model(model_name: str) -> dict:
    """测试单个模型，返回计时和结果信息。"""
    print(f"\n{'='*60}")
    print(f"  测试模型: {model_name}")
    print(f"{'='*60}")

    mem_before = get_nvidia_smi_mb()
    print(f"  测试前显存: {mem_before:.0f} MiB")

    # --- 加载模型 ---
    print(f"  加载模型中...", end="", flush=True)
    t_load_start = time.perf_counter()
    try:
        engine = OCREngine(model_name)
        engine.load()
    except Exception as e:
        print(f" 失败: {e}")
        return {
            "model": model_name,
            "status": "load_error",
            "error": str(e),
        }
    t_load = time.perf_counter() - t_load_start
    mem_after_load = get_nvidia_smi_mb()
    print(f" 完成 ({t_load:.1f}s, 显存 {mem_after_load:.0f} MiB)")

    # --- 推理 (warmup) ---
    print(f"  Warmup 推理...", end="", flush=True)
    t_warmup_start = time.perf_counter()
    try:
        result_warmup = engine.ocr(str(TEST_IMAGE))
    except Exception as e:
        print(f" 失败: {e}")
        engine.unload()
        force_cleanup()
        return {
            "model": model_name,
            "status": "inference_error",
            "error": str(e),
            "load_time_s": round(t_load, 2),
        }
    t_warmup = time.perf_counter() - t_warmup_start
    print(f" 完成 ({t_warmup:.1f}s)")

    # --- 正式推理 (计时) ---
    print(f"  正式推理...", end="", flush=True)
    t_infer_start = time.perf_counter()
    try:
        result = engine.ocr(str(TEST_IMAGE))
    except Exception as e:
        print(f" 失败: {e}")
        engine.unload()
        force_cleanup()
        return {
            "model": model_name,
            "status": "inference_error",
            "error": str(e),
            "load_time_s": round(t_load, 2),
            "warmup_time_s": round(t_warmup, 2),
        }
    t_infer = time.perf_counter() - t_infer_start
    mem_peak = get_nvidia_smi_mb()
    print(f" 完成 ({t_infer:.1f}s, 峰值显存 {mem_peak:.0f} MiB)")

    # 提取金额
    from benchmark import extract_amount
    amount = extract_amount(result.text)
    correct = amount is not None and abs(amount - EXPECTED_AMOUNT) < 0.01

    text_preview = result.text[:200].replace("\n", " ")
    print(f"  提取金额: ¥{amount}  {'✓' if correct else '✗'} (期望 ¥{EXPECTED_AMOUNT})")
    print(f"  OCR 文本预览: {text_preview}...")

    # --- 卸载模型 ---
    print(f"  卸载模型...", end="", flush=True)
    t_unload_start = time.perf_counter()
    engine.unload()
    del engine
    force_cleanup()
    t_unload = time.perf_counter() - t_unload_start
    time.sleep(1)  # 等待 GPU 回收
    mem_after_unload = get_nvidia_smi_mb()
    print(f" 完成 ({t_unload:.1f}s, 显存 {mem_after_unload:.0f} MiB)")

    return {
        "model": model_name,
        "status": "ok",
        "load_time_s": round(t_load, 2),
        "warmup_time_s": round(t_warmup, 2),
        "inference_time_s": round(t_infer, 2),
        "unload_time_s": round(t_unload, 2),
        "vram_loaded_mib": round(mem_after_load),
        "vram_peak_mib": round(mem_peak),
        "vram_after_unload_mib": round(mem_after_unload),
        "amount": amount,
        "correct": correct,
        "text_len": len(result.text),
    }


def print_summary(results: list[dict]):
    """打印汇总表格。"""
    print(f"\n\n{'='*100}")
    print(f"  速度测试汇总  —  测试图片: {TEST_IMAGE.name} (¥{EXPECTED_AMOUNT})")
    print(f"{'='*100}")

    header = f"{'模型':<16} {'状态':<6} {'加载(s)':>8} {'Warmup(s)':>10} {'推理(s)':>8} {'VRAM峰值':>10} {'金额':>8} {'正确':>4}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["status"] != "ok":
            print(f"{r['model']:<16} {'失败':<6} {r.get('error', '')[:50]}")
            continue
        print(
            f"{r['model']:<16} "
            f"{'OK':<6} "
            f"{r['load_time_s']:>7.1f}s "
            f"{r['warmup_time_s']:>9.1f}s "
            f"{r['inference_time_s']:>7.1f}s "
            f"{r['vram_peak_mib']:>8.0f}Mi "
            f"{'¥'+str(r['amount']) if r['amount'] else 'N/A':>8} "
            f"{'✓' if r['correct'] else '✗':>4}"
        )

    print(f"\n注: Warmup = 首次推理(含 CUDA kernel 编译等), 推理 = 第二次推理(稳定速度)")


def main():
    print(f"单张图片 OCR 速度测试")
    print(f"测试图片: {TEST_IMAGE}")
    print(f"待测模型: {', '.join(ALL_MODELS)}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    if not TEST_IMAGE.exists():
        print(f"错误: 测试图片不存在 {TEST_IMAGE}")
        return

    # 确保起始显存干净
    force_cleanup()
    baseline_mem = get_nvidia_smi_mb()
    print(f"基线显存: {baseline_mem:.0f} MiB")

    results = []
    for model_name in ALL_MODELS:
        r = test_single_model(model_name)
        results.append(r)

        # 确保显存恢复
        force_cleanup()
        time.sleep(2)
        mem_now = get_nvidia_smi_mb()
        if mem_now > baseline_mem + 500:
            print(f"  ⚠ 显存未完全恢复: {mem_now:.0f} MiB (基线 {baseline_mem:.0f} MiB)")

    print_summary(results)


if __name__ == "__main__":
    main()
