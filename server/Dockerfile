# Use Python 3.10 slim image for better compatibility
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgomp1 \
    gfortran \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools Cython

# Install numpy first (required for scikit-surprise build)
RUN pip install --no-cache-dir numpy==1.23.5

# Copy requirements and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir scikit-surprise==1.1.1 && \
    grep -v "scikit-surprise" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy the application code
COPY . .

# Make sure necessary directories exist
RUN mkdir -p e-commerce-models e-commerce-data

# Copy model and data files
COPY e-commerce-models/* e-commerce-models/
COPY e-commerce-data/* e-commerce-data/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]