#!/bin/bash
set -e

echo "Starting vLLM OpenAI server..."


export HOME=/tmp
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface
export VLLM_CACHE_ROOT=/tmp/vllm


mkdir -p /tmp/huggingface
mkdir -p /tmp/vllm

# Debug
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "HOME=$HOME"

# GPU diagnostics
nvidia-smi || echo "No GPU detected"

python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
EOF

# 🚀 Start server
exec python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3Guard-Gen-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype float16 \
    --max-model-len 256 \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-num-seqs 1  \
    --enforce-eager
