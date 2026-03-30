FROM vllm/vllm-openai:v0.8.2

ENV VLLM_USE_V1=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV HOME=/tmp
ENV HF_HOME=/tmp/huggingface
ENV VLLM_CACHE_ROOT=/tmp/vllm
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/huggingface /tmp/vllm /models && \
    chmod -R 777 /tmp/huggingface /tmp/vllm /models


RUN pip install --no-cache-dir --upgrade --force-reinstall \
    transformers==4.51.3

# Bake model
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

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
ENTRYPOINT ["/app/start.sh"]
#Fix-v1
