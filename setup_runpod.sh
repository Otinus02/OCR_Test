#!/bin/bash
# =============================================================
# RunPod Full Setup - OCR Benchmark Environment
# Installs all dependencies and starts vLLM server
# =============================================================

set -e

echo "=== OCR Benchmark - RunPod Setup ==="

# 1. Install Python dependencies for benchmark
echo "[1/5] Installing benchmark dependencies..."
pip install requests Pillow PyMuPDF pymupdf4llm markitdown

# NOTE: GLM-OCR requires transformers>=5.1.0, but vLLM pins transformers<5.
# Workaround: install vLLM first (pulls transformers 4.x), then overwrite with v5.
# pip doesn't re-check constraints of already-installed packages on sequential installs.
# Runtime is compatible since vLLM 0.14.0+. (https://github.com/vllm-project/vllm/issues/30466)
# This will be resolved once vLLM PR #30566 merges.

# 2. Install vLLM nightly (brings transformers 4.x)
echo "[2/5] Installing vLLM (nightly)..."
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# 3. Overwrite transformers to v5+ (GLM-OCR requirement)
echo "[3/5] Installing transformers v5 (overriding vLLM pin)..."
pip install git+https://github.com/huggingface/transformers.git

# 4. Clone repository
echo "[4/5] Setting up repository..."
if [ -d "/workspace/OCR_Test" ]; then
    echo "  Repository already exists, pulling latest..."
    cd /workspace/OCR_Test && git pull && cd -
else
    git clone https://github.com/Otinus02/OCR_Test.git /workspace/OCR_Test
fi
mkdir -p /workspace/OCR_Test/test_pdfs
mkdir -p /workspace/OCR_Test/results

# 5. Start vLLM server in background
echo "[5/5] Starting vLLM server (background)..."
nohup vllm serve zai-org/GLM-OCR \
    --allowed-local-media-path / \
    --port 8080 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 1}' \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    > /workspace/vllm.log 2>&1 &

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "vLLM server is starting in the background."
echo "  - Log: tail -f /workspace/vllm.log"
echo "  - Wait for 'Uvicorn running' before running benchmark"
echo ""
echo "Usage:"
echo "  cd /workspace/OCR_Test"
echo "  # Place PDFs in test_pdfs/"
echo "  python benchmark.py --pdf test_pdfs/your_file.pdf"
echo ""
echo "  # Run specific methods only:"
echo "  python benchmark.py --pdf test_pdfs/your_file.pdf --methods pymupdf4llm,markitdown"
echo ""
echo "  # View results:"
echo "  cat results/*/summary.txt"
echo ""
