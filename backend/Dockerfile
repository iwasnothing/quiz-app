# Use NVIDIA CUDA base image for GPU support on Spark DGX
# Using devel variant for faiss-gpu compilation
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip (force overwrite if they exist)
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python3.11 -m pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install faiss-gpu

# Note: App code will be mounted as a volume in docker-compose for development
# This allows code changes without rebuilding the image
# COPY . .

# Run with uvicorn (FastAPI server) with reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]