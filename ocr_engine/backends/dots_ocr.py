"""dots.ocr backend - multilingual document layout parsing VLM."""

import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from ..base import ModelConfig, OCRBackend, OCRResult, InferenceMode
from . import register_backend

# Simplified OCR-only prompt (no layout/bbox, just extract text)
DOTS_OCR_PROMPT = """\
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""


@register_backend("dots-ocr")
class DotsOCRBackend(OCRBackend):
    """
    dots.ocr backend (rednote-hilab/dots.ocr).

    1.7B VLM for multilingual document layout parsing.
    Uses standard transformers AutoModelForCausalLM (no dots.ocr package needed).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    @property
    def name(self) -> str:
        return "dots-ocr"

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode.OFFLINE

    def load(self) -> None:
        if self._loaded:
            return

        import os
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            model_path = self.config.model_id  # Download from HuggingFace
        dtype = self._get_torch_dtype()

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()

    def ocr(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str = "",
    ) -> OCRResult:
        self.ensure_loaded()
        start_time = time.perf_counter()

        try:
            from qwen_vl_utils import process_vision_info

            img = self._load_image(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": DOTS_OCR_PROMPT},
                    ],
                }
            ]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Try to extract plain text from JSON layout output
            text = self._extract_text_from_layout(output_text)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                text=text,
                model_name=self.name,
                inference_mode=self.inference_mode,
                tokens_generated=len(generated_ids_trimmed[0]),
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return OCRResult(
                text="",
                model_name=self.name,
                inference_mode=self.inference_mode,
                processing_time_ms=elapsed_ms,
                error=str(e),
            )

    @staticmethod
    def _extract_text_from_layout(raw_output: str) -> str:
        """Extract plain text from dots.ocr JSON layout output.

        The model outputs a JSON array of layout elements with 'text' fields.
        Concatenate all text fields in reading order.
        Falls back to raw output if JSON parsing fails.
        """
        import json
        try:
            elements = json.loads(raw_output)
            if isinstance(elements, list):
                parts = []
                for el in elements:
                    if isinstance(el, dict) and "text" in el:
                        parts.append(el["text"])
                if parts:
                    return "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass
        return raw_output
