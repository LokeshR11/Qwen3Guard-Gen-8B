#!/bin/bash
set -e

echo "Starting vLLM server for Qwen3Guard-Gen-8B..."
echo "Model path: /models/Qwen3Guard-Gen-8B"
echo "Context length: 2048 | GPU utilization: 0.80"

# GPU diagnostics
nvidia-smi || echo "nvidia-smi not available or no GPU visible"
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
" || echo "torch CUDA check failed"

exec python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3Guard-Gen-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype float16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.75 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-num-seqs 1
