# backend/Dockerfile
FROM python:3.11-slim

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend /app/backend

# Install Python deps
RUN pip install --no-cache-dir -r backend/requirements.txt

# Build FAISS index at image build time if gallery exists
RUN python backend/scripts/build_index.py || true

EXPOSE 5000
CMD ["python", "backend/app.py"]

