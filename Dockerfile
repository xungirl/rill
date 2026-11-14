# CPU-only image for the ember server. Linux torch comes from the PyTorch CPU index
# (see [tool.uv.sources] in pyproject.toml), so no CUDA wheels are downloaded.
#
#   docker build -t ember:cpu .
#   docker run --rm -p 8100:8100 -v ~/.cache/huggingface:/models:ro ember:cpu
#
# The model is NOT baked into the image: mount a local Hugging Face cache that contains
# Qwen/Qwen3-0.6B (ember never downloads weights).
FROM python:3.15-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" \
    HF_HUB_CACHE=/models/hub
WORKDIR /app

# dependency layer (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY ember ./ember
COPY README.md ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home ember
USER ember
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health')"
ENTRYPOINT ["python", "-m", "ember.server", "--device", "cpu", "--host", "0.0.0.0", "--port", "8100"]
