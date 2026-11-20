FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/cache/huggingface \
    TORCH_HOME=/models/cache/torch \
    XDG_CACHE_HOME=/models/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ARG GIGAAM_REF=main
RUN git clone https://github.com/salute-developers/GigaAM.git /opt/GigaAM \
    && cd /opt/GigaAM && git checkout "$GIGAAM_REF"

# CPU PyTorch, no NVIDIA CUDA packages.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && python -m pip install --no-cache-dir '/opt/GigaAM[torch,longform]' \
       fastapi 'uvicorn[standard]' python-multipart

WORKDIR /app
COPY app.py /app/app.py
RUN mkdir -p /models/cache
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
