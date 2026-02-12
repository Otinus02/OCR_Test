#!/bin/bash
# =============================================================
# GLM-OCR vLLM Server Setup Script
# Run this on your RunPod/Vast.ai GPU instance
# =============================================================

set -e

echo "=== GLM-OCR vLLM Server Setup ==="

# NOTE: GLM-OCR requires transformers>=5.1.0, but vLLM pins transformers<5.
# Workaround: install vLLM first (pulls transformers 4.x), then overwrite with v5.
# pip doesn't re-check constraints of already-installed packages on sequential installs.
# Runtime is compatible since vLLM 0.14.0+. (https://github.com/vllm-project/vllm/issues/30466)
# This will be resolved once vLLM PR #30566 merges.

# Step 1: Install vLLM nightly (brings transformers 4.x)
echo "[1/3] Installing vLLM..."
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Step 2: Overwrite transformers to v5+ (GLM-OCR requirement)
echo "[2/3] Installing transformers v5 (overriding vLLM pin)..."
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
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 1}' \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1
