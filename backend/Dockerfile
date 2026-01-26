# Use NVIDIA CUDA base image for GPU support on Spark DGX
# Using devel variant for faiss-gpu compilation
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Install Python 3.11 via conda and pip
RUN conda install -y python=3.11 pip && \
    pip install --upgrade pip

# Install faiss-gpu via conda (not available on PyPI)
RUN conda install -y -c conda-forge faiss-gpu

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: App code will be mounted as a volume in docker-compose for development
# This allows code changes without rebuilding the image
# COPY . .

# Run with uvicorn (FastAPI server) with reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]