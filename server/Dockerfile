# Build stage
FROM python:3.10-alpine as builder

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    python3-dev \
    openblas-dev \
    linux-headers \
    gcc \
    g++ \
    gfortran \
    musl-dev \
    lapack-dev

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir -U pip setuptools wheel Cython

# Install numpy first (required for scikit-surprise)
RUN pip install --no-cache-dir numpy==1.23.5

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir scikit-surprise==1.1.1 && \
    grep -v "scikit-surprise" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Compile Python files
RUN python -m compileall /opt/venv

# Final stage
FROM python:3.10-alpine

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONOPTIMIZE=2 \
    PYTHONDONTWRITEBYTECODE=1

# Install only required runtime dependencies
RUN apk add --no-cache libstdc++ openblas libgomp lapack

# Set working directory
WORKDIR /app

# Copy only necessary files and create cache directories
COPY ./api ./api
COPY ./models ./models
COPY ./services ./services
COPY ./utils ./utils
COPY main.py config.py ./

# Create model and data directories and copy files
COPY e-commerce-models ./e-commerce-models
COPY e-commerce-data ./e-commerce-data

# Create cache directory for better performance
RUN mkdir -p /app/cache && \
    chmod -R 777 /app/cache

# Add non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
RUN chown -R appuser:appgroup /app
USER appuser

# Expose port
EXPOSE 8000

# Set resource limits and runtime optimizations
ENV UVICORN_WORKERS=4 \
    UVICORN_LOOP=uvloop \
    UVICORN_HTTP=httptools \
    PYTHONHASHSEED=random

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Command to run the application with optimized settings
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--proxy-headers", \
     "--no-access-log"]