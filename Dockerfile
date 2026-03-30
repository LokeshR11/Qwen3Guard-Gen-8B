FROM vllm/vllm-openai:v0.7.3


ENV HOME=/tmp
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV VLLM_CACHE_ROOT=/tmp/vllm
ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO

# Install transformers (required for Qwen)
RUN pip install --no-cache-dir "transformers>=4.51.0"


RUN mkdir -p /tmp/huggingface /tmp/vllm /models

# ✅ Bake model
RUN python3 - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3Guard-Gen-8B",
    local_dir="/models/Qwen3Guard-Gen-8B",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    ignore_patterns=["*.msgpack"]
)

print("✅ Model baked successfully!")
EOF

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080

ENTRYPOINT ["/app/start.sh"]
#Fix-v4
