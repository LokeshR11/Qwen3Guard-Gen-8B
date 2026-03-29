FROM vllm/vllm-openai:latest

# Environment
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO

# Transformers for Qwen3 architecture
RUN pip install --no-cache-dir "transformers>=4.51.0"

# Bake the model
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

# Clean cache
RUN rm -rf /root/.cache/huggingface

# Copy and prepare startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080


ENTRYPOINT ["/app/start.sh"]
