#!/bin/bash
# =============================================================
# GLM-OCR vLLM Server Setup Script
# Run this on your RunPod/Vast.ai GPU instance
# =============================================================

set -e

echo "=== GLM-OCR vLLM Server Setup ==="

# Install vLLM (nightly for GLM-OCR support)
echo "[1/3] Installing vLLM..."
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

# Install transformers from source (required for GLM-OCR)
echo "[2/3] Installing transformers (latest)..."
pip install git+https://github.com/huggingface/transformers.git

# Start vLLM server
echo "[3/3] Starting vLLM server with GLM-OCR..."
echo ""
echo "Server will be available at http://0.0.0.0:8080"
echo "Use the external IP of this instance to connect from your local machine."
echo ""

vllm serve zai-org/GLM-OCR \
    --allowed-local-media-path / \
    --port 8080 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
