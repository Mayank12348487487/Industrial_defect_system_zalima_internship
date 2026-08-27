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
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY onnx_inference.py .
COPY video_stream.py .
COPY best_industrial_defect.onnx .

# Copy dataset structure for directory simulation fallback
COPY data/data.yaml ./data/data.yaml
COPY data/images/val/ ./data/images/val/

# Copy FastAPI app structure
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Start app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
