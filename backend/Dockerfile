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

# Install miniconda for faiss-gpu (not available on PyPI)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_ARCH="x86_64"; \
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        MINICONDA_ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    apt-get update && apt-get install -y wget && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Install faiss-gpu via conda
RUN conda install -y -c conda-forge faiss-gpu && \
    conda clean -afy

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: App code will be mounted as a volume in docker-compose for development
# This allows code changes without rebuilding the image
# COPY . .

# Run with uvicorn (FastAPI server) with reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]