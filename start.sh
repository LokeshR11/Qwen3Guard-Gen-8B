#!/bin/bash
set -e

echo " Starting vLLM server for Qwen3Guard-Gen-8B..."
echo " Model path: /models/Qwen3Guard-Gen-8B"
echo " Context length: 2048 | GPU utilization: 0.80"

# Optional: show GPU status for debugging
nvidia-smi || echo "nvidia-smi not available or no GPU visible"

# Start vLLM using the modern CLI (recommended)
exec vllm serve /models/Qwen3Guard-Gen-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.80 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enforce-eager \
    --max-num-seqs 4
