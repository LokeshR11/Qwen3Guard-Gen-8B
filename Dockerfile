FROM vllm/vllm-openai:v0.8.2

# -------------------------------
# Environment
# -------------------------------
ENV VLLM_USE_V1=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV HOME=/tmp
ENV HF_HOME=/tmp/huggingface
ENV VLLM_CACHE_ROOT=/tmp/vllm
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# -------------------------------
# Create required directories
# -------------------------------
RUN mkdir -p /tmp/huggingface /tmp/vllm /models && \
    chmod -R 777 /tmp/huggingface /tmp/vllm /models

# -------------------------------
# FIX: Use EXACT compatible transformers
# -------------------------------
RUN pip uninstall -y transformers && \
    pip install --no-cache-dir transformers==4.51.0

# -------------------------------
# Bake model into image
# -------------------------------
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

print("Model baked successfully!")
EOF

# -------------------------------
# Copy startup script
# -------------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# -------------------------------
# Expose port
# -------------------------------
EXPOSE 8080

# -------------------------------
# Start server
# -------------------------------
ENTRYPOINT ["/app/start.sh"]
