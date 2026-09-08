FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and models
COPY onnx_inference.py .
COPY video_stream.py .
COPY best_industrial_defect.onnx .
COPY industry_video.mp4 .

# Setup directory structure and data config
RUN mkdir -p data/images/val data/uploads data/output_stream
COPY data/data.yaml ./data/data.yaml

# Copy FastAPI app structure
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Container Health Check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
