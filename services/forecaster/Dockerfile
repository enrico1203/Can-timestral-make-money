FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# Pinned commit of mistralai/timestral for reproducible builds
ARG TIMESTRAL_SHA=a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
RUN pip install "git+https://github.com/mistralai/timestral.git@${TIMESTRAL_SHA}"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
ENV TIMESTRAL_SHA=${TIMESTRAL_SHA}
WORKDIR /app
COPY app.py /app/app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
