
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build, not runtime
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

CMD uvicorn app:app --host 0.0.0.0 --port $PORTs