# Use python-slim for a smaller ARM64 footprint
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg (Postgres driver)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: App code will be mounted as a volume in docker-compose for development
# This allows code changes without rebuilding the image
# COPY . .

# Run with uvicorn (FastAPI server) with reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]