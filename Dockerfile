# Base Image - Python 3.10 slim
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
     git \
     curl \
     && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.yml .
COPY main.py .
COPY download_models.py .
COPY pipeline/ ./pipeline/
COPY retrieval/ ./retrieval/
COPY generation/ ./generation/
COPY tracking/ ./tracking/
COPY evaluation/ ./evaluation/
COPY serving/ ./serving/
COPY utils/ ./utils/
COPY docs/ ./docs/

# Environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV WEIGHTS_PATH=/app/weights

# FastAPI port
EXPOSE 8000

# Default command — start serving
CMD ["python", "main.py", "serve"]