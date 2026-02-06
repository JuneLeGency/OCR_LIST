# GLM-OCR Makefile
# 使用 vLLM 部署 GLM-OCR 模型的常用命令

# Python 解释器路径（使用 venv 中的 python 避免依赖被覆盖）
PYTHON := .venv/bin/python
UV := uv

# 服务配置
PORT := 9090
HOST := 0.0.0.0

# 默认目标
.DEFAULT_GOAL := help

# ============================================================================
# 环境管理
# ============================================================================

.PHONY: install
install: ## 初始化环境并安装依赖
	$(UV) sync
	@echo "安装 transformers 开发版（GLM-OCR 需要）..."
	@if [ -d "/tmp/transformers" ]; then \
		cd /tmp/transformers && git pull; \
	else \
		git clone --depth=1 https://github.com/huggingface/transformers.git /tmp/transformers; \
	fi
	.venv/bin/pip install --no-deps /tmp/transformers
	.venv/bin/pip install --upgrade huggingface-hub
	@echo "✓ 安装完成"