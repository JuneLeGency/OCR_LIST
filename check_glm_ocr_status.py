#!/usr/bin/env python3
"""检查 GLM-OCR 服务器状态"""

import sys
import requests


def check_server_status(base_url="http://localhost:9090"):
    """检查服务器是否运行"""
    try:
        # 检查健康状态
        health_url = f"{base_url}/health"
        response = requests.get(health_url, timeout=2)

        if response.status_code == 200:
            print("✓ GLM-OCR 服务器运行正常")
            print(f"  地址: {base_url}")

            # 尝试获取模型信息
            try:
                models_url = f"{base_url}/v1/models"
                models_response = requests.get(models_url, timeout=2)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    if "data" in models_data and models_data["data"]:
                        print(f"  可用模型: {models_data['data'][0].get('id', 'unknown')}")
            except Exception:
                pass

            return True
        else:
            print(f"✗ 服务器响应异常: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ GLM-OCR 服务器未运行")
        print(f"  无法连接到: {base_url}")
        print("\n请先启动服务器:")
        print("  python start_glm_ocr_server.py")
        return False

    except Exception as e:
        print(f"✗ 检查失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("检查 GLM-OCR 服务器状态")
    print("=" * 60)
    print()

    is_running = check_server_status()

    print()
    if is_running:
        print("准备就绪！可以开始使用 GLM-OCR 进行识别。")
        print("\n测试命令:")
        print("  python test_glm_ocr_single.py ./inputs/test.jpg")
    else:
        print("需要先启动 GLM-OCR 服务器。")

    sys.exit(0 if is_running else 1)
