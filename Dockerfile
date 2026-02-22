# ─── Stage 1: download model weights ─────────────────────────────────────────
FROM python:3.12-slim AS model-downloader

RUN pip install --no-cache-dir huggingface_hub

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download all model files into /model
RUN python - <<'EOF'
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
    local_dir="/model",
    token=os.environ.get("HF_TOKEN") or None,
    ignore_patterns=["*.md", ".gitattributes", "original/*"],
)
EOF

# ─── Stage 2: runtime image ───────────────────────────────────────────────────
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights from the downloader stage
COPY --from=model-downloader /model /model

COPY src/handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
