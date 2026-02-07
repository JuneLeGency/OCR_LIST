"""Gradio Web UI for OCR model comparison."""

import gc
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from .engine import OCREngine
from .config import SUPPORTED_MODELS, VLLM_CONFIGS


def _format_results_markdown(results: list[dict]) -> str:
    """Format results as markdown for display."""
    if not results:
        return ""

    parts = []
    for r in results:
        status = "ERROR" if r.get("error") else "OK"
        time_str = f'{r["time_ms"]:.0f}ms' if r.get("time_ms") else "N/A"
        header = f'### {r["model"]}  `{time_str}`  `{status}`'
        parts.append(header)

        if r.get("error"):
            parts.append(f'\n> {r["error"]}\n')
        else:
            parts.append(f'\n```\n{r["text"]}\n```\n')

    return "\n".join(parts)


def _format_running_status(completed: list[dict], current_model: str, remaining: int) -> str:
    """Format status while running."""
    result_md = _format_results_markdown(completed)
    status = f"⏳ **正在运行: {current_model}** (剩余 {remaining} 个模型)\n\n---\n\n"
    return status + result_md


def run_comparison(image, selected_models: list[str], prompt: str):
    """Run OCR on selected models and yield results progressively."""
    if image is None:
        yield "请先上传图片。"
        return

    if not selected_models:
        yield "请至少选择一个模型。"
        return

    results = []
    total = len(selected_models)

    for i, model_name in enumerate(selected_models):
        remaining = total - i - 1
        yield _format_running_status(results, model_name, remaining)

        entry = {"model": model_name, "text": "", "time_ms": 0, "error": None}
        engine = None
        try:
            engine = OCREngine(model_name)
            t0 = time.time()
            result = engine.ocr(image, prompt)
            elapsed = (time.time() - t0) * 1000

            entry["time_ms"] = elapsed
            if result.error:
                entry["error"] = result.error
            else:
                entry["text"] = result.text
        except Exception as e:
            entry["error"] = str(e)
        finally:
            if engine is not None:
                engine.unload()
            del engine
            gc.collect()
            torch.cuda.empty_cache()

        results.append(entry)

    # Final output
    summary_parts = ["## 比较结果\n"]
    summary_parts.append("| 模型 | 耗时 | 状态 |")
    summary_parts.append("|------|------|------|")
    for r in results:
        status = "❌ Error" if r.get("error") else "✅ OK"
        time_str = f'{r["time_ms"]:.0f}ms' if r.get("time_ms") else "N/A"
        summary_parts.append(f'| {r["model"]} | {time_str} | {status} |')

    summary_parts.append("\n---\n")
    summary_md = "\n".join(summary_parts)
    detail_md = _format_results_markdown(results)

    yield summary_md + detail_md


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    all_models = list(SUPPORTED_MODELS.keys())
    vllm_models = list(VLLM_CONFIGS.keys())

    with gr.Blocks(title="OCR Model Comparison") as app:
        gr.Markdown("# OCR 多模型比较\n上传图片，选择模型，对比 OCR 结果。")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传图片")
                prompt_input = gr.Textbox(
                    value="请识别图片中的所有文字内容，保持原有格式。",
                    label="Prompt",
                    lines=2,
                )
                with gr.Accordion("离线模型 (本地推理)", open=True):
                    offline_models = gr.CheckboxGroup(
                        choices=all_models,
                        value=["rapidocr"],
                        label="选择模型",
                    )
                if vllm_models:
                    with gr.Accordion("vLLM 模型 (服务端推理)", open=False):
                        vllm_model_select = gr.CheckboxGroup(
                            choices=vllm_models,
                            value=[],
                            label="选择模型",
                        )
                else:
                    vllm_model_select = gr.State([])

                run_btn = gr.Button("🚀 开始比较", variant="primary", size="lg")

            with gr.Column(scale=2):
                output = gr.Markdown(
                    value="上传图片并点击「开始比较」查看结果。",
                    label="比较结果",
                )

        def on_run(image, offline, vllm_sel, prompt):
            selected = list(offline or []) + list(vllm_sel or [])
            yield from run_comparison(image, selected, prompt)

        run_btn.click(
            fn=on_run,
            inputs=[image_input, offline_models, vllm_model_select, prompt_input],
            outputs=output,
        )

    return app


def run_web(host: str = "0.0.0.0", port: int = 9090):
    """Launch the Gradio web UI."""
    app = create_app()
    app.launch(server_name=host, server_port=port, theme=gr.themes.Soft())
