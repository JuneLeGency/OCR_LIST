#!/usr/bin/env python3
"""
Simple test of HunYuan OCR to extract amount
"""

from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from PIL import Image
import torch
import re
from pathlib import Path

def extract_amount_from_text(text: str) -> float:
    """Extract amount from OCR text"""
    if not text:
        return None

    patterns = [
        r'（小写）\s*([0-9]+\.?[0-9]*)',
        r'\(小写\)\s*([0-9]+\.?[0-9]*)',
        r'小写[）:：]\s*([0-9]+\.?[0-9]*)',
        r'合计[：:]\s*([0-9]+\.?[0-9]*)',
        r'总计[：:]\s*([0-9]+\.?[0-9]*)',
        r'金额[：:]\s*([0-9]+\.?[0-9]*)',
        r'金额\s*([0-9]+\.?[0-9]*)',
        r'￥\s*([0-9]+\.?[0-9]*)',
        r'\$\s*([0-9]+\.?[0-9]*)',
        r'([0-9]+\.?[0-9]*)\s*元',
        r'([0-9]+\.?[0-9]*)\s*RMB',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])

    numbers = re.findall(r'([0-9]+\.?[0-9]*)', text)
    if numbers:
        return float(max(numbers, key=lambda x: float(x)))

    return None

# Test images
test_images = [
    "./inputs/25337000000272477597.jpg",
    "./inputs/收费票据 0607575447.jpg",
]

print("Testing HunYuan OCR with uv run")
print("=" * 80)

try:
    print("Loading processor...", end=" ", flush=True)
    processor = AutoProcessor.from_pretrained("tencent/HunyuanOCR", use_fast=False)
    print("✓")

    print("Loading model...", end=" ", flush=True)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        "tencent/HunyuanOCR",
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓")

    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"\n⚠️ {img_path} not found, skipping")
            continue

        print(f"\nProcessing: {img_path}")
        print("-" * 80)

        try:
            image_inputs = Image.open(img_path)

            messages1 = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": "检测并识别图片中的文字，特别是金额。"},
                    ],
                }
            ]

            messages = [messages1]
            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]

            print("Preparing inputs...", end=" ", flush=True)
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            print("✓")

            print("Generating...", end=" ", flush=True)
            device = next(model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

            print("✓")

            if "input_ids" in inputs:
                input_ids = inputs.input_ids
            else:
                input_ids = inputs.inputs if hasattr(inputs, 'inputs') else inputs.input_ids

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            full_text = output_texts[0] if output_texts else ''

            print(f"\nRecognized text (first 500 chars):")
            print("-" * 80)
            print(full_text[:500])
            print("-" * 80)

            amount = extract_amount_from_text(full_text)
            print(f"\nExtracted amount: ¥{amount:.2f}" if amount else "Failed to extract")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n✗ GPU out of memory: {e}")
            else:
                print(f"\n✗ RuntimeError: {e}")
        except Exception as e:
            print(f"\n✗ Error: {e}")

except Exception as e:
    print(f"\n✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
