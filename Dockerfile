FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO

RUN mkdir -p /models/Qwen3Guard-Gen-8B && \
    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3Guard-Gen-8B",
    local_dir="/models/Qwen3Guard-Gen-8B",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    ignore_patterns=["*.bin", "*.ot", "*.msgpack"]
)
print("✅ Model baked successfully!")
EOF

RUN rm -rf /root/.cache/huggingface

EXPOSE 8080

CMD ["vllm", "serve", \
     "/models/Qwen3Guard-Gen-8B", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--dtype", "auto", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.85", \
     "--tensor-parallel-size", "1", \
     "--trust-remote-code", \
     "--enforce-eager", \
     "--max-num-seqs", "4"]
