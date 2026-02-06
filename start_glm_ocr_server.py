#!/usr/bin/env python3
"""
启动 vLLM 服务器运行 GLM-OCR 模型
需要先安装 vllm: pip install vllm
"""

import os
import subprocess
import sys

# 使用 ModelScope 下载模型
os.environ["VLLM_USE_MODELSCOPE"] = "true"


def main():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "ZhipuAI/GLM-OCR",
        "--served-model-name", "glm-ocr",
        "--host", "0.0.0.0",
        "--port", "9090",
        "--trust-remote-code",
        "--allowed-local-media-path", "/",
        "--max-model-len", "8192",
    ]

    print("=" * 80)
    print("启动 GLM-OCR 服务 (使用 ModelScope)")
    print("=" * 80)
    print(f"命令: {' '.join(cmd)}")
    print("\n服务地址: http://localhost:9090")
    print("API 端点: http://localhost:9090/v1")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 80)
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n服务已停止")


if __name__ == "__main__":
    main()
