#!/usr/bin/env python3
"""
测试单张图片使用 GLM-OCR 识别
直接加载模型，无需启动服务器
"""

import sys
from pathlib import Path
from glm_ocr import process_image_with_glm_ocr_direct


def main():
    if len(sys.argv) < 2:
        print("用法: python test_glm_ocr_single.py <image_path> [prompt]")
        print("\n支持的 prompt 模式:")
        print("  - Text Recognition: (默认) 文本识别")
        print("  - Document Parse: 文档解析")
        print("  - Formula Recognition: 公式识别")
        print("  - Table Recognition: 表格识别")
        print("  - Information Extraction: 信息提取")
        print("\n示例:")
        print('  uv run python test_glm_ocr_single.py ./inputs/test.jpg')
        print('  uv run python test_glm_ocr_single.py ./inputs/test.jpg "Document Parse:"')
        sys.exit(1)

    img_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Text Recognition:"

    if not Path(img_path).exists():
        print(f"错误: 图片文件不存在: {img_path}")
        sys.exit(1)

    print("=" * 80)
    print("GLM-OCR 测试 (直接加载模型)")
    print("=" * 80)
    print(f"图片路径: {img_path}")
    print(f"识别模式: {prompt}")
    print("=" * 80)
    print()

    try:
        print("正在加载模型...")
        result = process_image_with_glm_ocr_direct(img_path, prompt=prompt)

        print("\n识别结果:")
        print("-" * 80)
        print(result)
        print("-" * 80)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
