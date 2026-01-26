# Use NVIDIA PyTorch base image for GPU support on Spark DGX
# This image includes Python 3.12, CUDA, and conda pre-installed
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# Install system dependencies for psycopg (Postgres driver)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install faiss-gpu via conda (not available on PyPI)
# Note: The base image already has conda in PATH
RUN conda install -y -c conda-forge faiss-gpu

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: App code will be mounted as a volume in docker-compose for development
# This allows code changes without rebuilding the image
# COPY . .

# Run with uvicorn (FastAPI server) with reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]