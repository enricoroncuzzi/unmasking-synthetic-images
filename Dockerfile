# Stage 1: build wheels
FROM python:3.10-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY demo/requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Stage 2: runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY demo/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY models/ ./models/
COPY demo/ ./demo/

ENV DEVICE=cpu
ENV STRATEGY=logit
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 7860 8000

CMD ["python", "-m", "demo.app"]
