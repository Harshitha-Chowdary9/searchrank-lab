# Runtime image for the search API (retrieval + reranking only, no training).
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Bake the public WANDS dataset in so the container is self-contained.
RUN python -c "from src.data import download_data; download_data('data')"

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
