# Local ONNX CPU image for Windows/WSL and ordinary x86_64 Linux hosts.
FROM python:3.11-slim-bookworm

ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    FACE_API_RUNTIME=onnx \
    FACE_API_RETINAFACE_PROVIDER=cpu

WORKDIR /app

# Docker Desktop's proxy may return a 502 for Debian's HTTP endpoints. Use a
# configurable HTTPS mirror and retry transient package-index failures.
RUN sed -ri \
        "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g; \
         s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Retries "5";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==0.12.17
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY face_api ./face_api
COPY face_core ./face_core
COPY scripts ./scripts
COPY run_demo.py ./

RUN groupadd --gid 1000 faceapi \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin faceapi \
    && mkdir -p /app/data/face_api /app/logs \
    && chown -R faceapi:faceapi /app/data /app/logs

ENV PATH=/app/.venv/bin:$PATH
USER faceapi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)" || exit 1

CMD ["python", "-m", "face_api.main", "--runtime", "onnx", "--provider", "cpu", "--retinaface-provider", "cpu", "--host", "0.0.0.0", "--port", "8000", "--performance-log-enabled", "--performance-log-file", "logs/onnx_performance.jsonl"]
