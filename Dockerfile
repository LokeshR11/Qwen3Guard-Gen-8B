FROM vllm/vllm-openai:latest

# Environment
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO

# Required for Qwen3 architecture
RUN pip install --no-cache-dir "transformers>=4.51.0"

# Bake the model into the image
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
print("Model baked successfully")
EOF

# Clean temporary HF cache
RUN rm -rf /root/.cache/huggingface

EXPOSE 8080

# Clear base image entrypoint — critical fix
ENTRYPOINT []

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/Qwen3Guard-Gen-8B", "--host", "0.0.0.0", "--port", "8080", "--dtype", "float16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.80", "--tensor-parallel-size", "1", "--trust-remote-code", "--enforce-eager", "--max-num-seqs", "2"]
