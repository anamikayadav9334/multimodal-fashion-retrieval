# Hugging Face Space (Docker SDK) - serves the FastAPI retriever on port 7860.
# Only query-time models are needed (FashionSigLIP, CLIP ViT-B/32, bge-small); they are
# downloaded at build time so the Space starts fast. Index + images are prebuilt locally.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir -e ".[ml,api]"

USER user
# warm the model cache (query-time encoders only)
RUN python -c "import open_clip; from sentence_transformers import SentenceTransformer; \
open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP'); \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=user configs ./configs
COPY --chown=user api ./api
COPY --chown=user eval ./eval
COPY --chown=user data ./data
COPY --chown=user artifacts ./artifacts

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
