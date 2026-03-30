#!/bin/bash
set -e

echo "Starting vLLM OpenAI server..."

# -------------------------------
# Environment
# -------------------------------
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HOME=/tmp
export HF_HOME=/tmp/huggingface
export VLLM_CACHE_ROOT=/tmp/vllm

mkdir -p /tmp/huggingface /tmp/vllm

# -------------------------------
# GPU Debug Info
# -------------------------------
nvidia-smi || echo "No GPU detected"

python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
EOF

# -------------------------------
# Start vLLM server
# -------------------------------
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3Guard-Gen-8B \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype float16 \
  --max-model-len 256 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-num-seqs 1 \
  --enforce-eager
