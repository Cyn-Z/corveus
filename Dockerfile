FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
COPY analyst/requirements.txt /app/requirements.txt
# Install CPU-only PyTorch inside the image.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch
RUN uv pip install --system -r /app/requirements.txt
COPY analyst/http_server.py analyst/orchestrator.py analyst/analyst.py analyst/config.py /app/
COPY analyst/skills /app/skills
COPY analyst/web /app/web 
COPY models/all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2
# Disables huggin face telemetry for offline builds
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
CMD ["python", "http_server.py"]

