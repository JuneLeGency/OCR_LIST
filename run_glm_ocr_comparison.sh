#!/bin/bash
# GLM-OCR 快速对比测试脚本

echo "=========================================="
echo "GLM-OCR 对比测试"
echo "=========================================="
echo ""

# 检查服务器状态
echo "1. 检查 GLM-OCR 服务器状态..."
uv run python check_glm_ocr_status.py
if [ $? -ne 0 ]; then
    echo ""
    echo "请在另一个终端启动 GLM-OCR 服务器："
    echo "  cd ../glm-ocr"
    echo "  make serve"
    echo ""
    echo "或者："
    echo "  python start_glm_ocr_server.py"
    exit 1
fi

echo ""
echo "2. 运行 GLM-OCR 对比测试..."
echo ""

# 运行对比测试（限制10张图片）
uv run python run_all_models_separate.py --models rapidocr glmocr --limit 10

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  ls -lh ocr_results/"
echo ""
