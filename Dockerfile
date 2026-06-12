FROM node:22-slim AS web
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY mold_inspection ./mold_inspection
COPY config ./config
RUN pip install --no-cache-dir -e ".[cloud,vision]"
COPY --from=web /app/web/dist ./web/dist
CMD ["uvicorn", "mold_inspection.cloud.app:app", "--host", "0.0.0.0", "--port", "8080"]
