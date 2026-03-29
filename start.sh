#!/bin/bash
set -e

echo "Starting vLLM OpenAI server..."

# Ensure cache dirs exist (CRITICAL FIX)
mkdir -p /root/.cache/huggingface
mkdir -p /tmp/vllm

# GPU diagnostics
nvidia-smi || echo "No GPU detected"

python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
EOF


exec python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3Guard-Gen-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-num-seqs 1
