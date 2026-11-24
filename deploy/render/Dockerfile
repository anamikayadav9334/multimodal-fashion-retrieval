# Render free tier (512 MB): torch-free API with int8 ONNX query encoders.
# Index + encoders are pulled from the public HF model repo at build time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 ENCODER_RUNTIME=onnx TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
WORKDIR /app

RUN pip install --no-cache-dir onnxruntime tokenizers numpy "qdrant-client>=1.12" "fastapi>=0.110" \
    "uvicorn>=0.29" pyyaml python-dotenv requests ftfy huggingface_hub pillow tqdm

ARG ASSETS_REPO=YellowBlue12/fashion-retrieval-assets
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${ASSETS_REPO}', local_dir='artifacts')"

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY api ./api
COPY eval/queries.yaml eval/results.json ./eval/

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
