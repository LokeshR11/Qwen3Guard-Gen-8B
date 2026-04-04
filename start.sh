#!/bin/bash
set -ex  

echo " Starting vLLM OpenAI server with FULL DEBUG..."

# =========================================================
# Environment Setup
# =========================================================
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=none

export HOME=/tmp
export HF_HOME=/tmp/huggingface
export VLLM_CACHE_ROOT=/tmp/vllm


export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /tmp/huggingface /tmp/vllm

echo " Environment variables set"

# =========================================================
# GPU DEBUG INFO (BEFORE MODEL LOAD)
# =========================================================
echo "================ GPU STATUS BEFORE MODEL LOAD ================"
nvidia-smi || echo " No GPU detected"

python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total VRAM:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
    print("Allocated VRAM:", torch.cuda.memory_allocated() // 1024**3, "GB")
    print("Reserved VRAM:", torch.cuda.memory_reserved() // 1024**3, "GB")
EOF

echo "=============================================================="

# =========================================================
# Start vLLM Server (WITH FULL LOGGING)
# =========================================================
echo " Launching vLLM..."

exec python3 -u -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3Guard-Gen-8B \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype float16 \
  --max-model-len 256 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-num-seqs 1 \
  --enforce-eager \
  --log-level debug \
  --engine-use-ray false \
  --distributed-executor-backend mp \
  --disable-custom-all-reduce
  2>&1 | tee /tmp/vllm_debug.log
