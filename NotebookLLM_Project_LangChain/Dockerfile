FROM python:3.11-slim

# Tesseract for OCR. libgl/glib are needed by PyMuPDF/Pillow on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/opt/models DATA_DIR=/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding and reranker models at build time, so the container runs offline.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}')" \
 && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('${RERANK_MODEL}')"

COPY app ./app
COPY eval ./eval

RUN useradd -m app && mkdir -p /data && chown -R app /data /opt/models
USER app
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:5000/health || exit 1

# One worker: the vector index lives in memory. Threads handle concurrent requests.
CMD ["gunicorn", "-w", "1", "--threads", "4", "-t", "180", "-b", "0.0.0.0:5000", "app.main:app"]
