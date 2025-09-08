# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements-exact.txt .

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir -r requirements-exact.txt

# Copy the package
COPY . .

# Install the package
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-c", "from risk_pipeline import Config, DualPipeline; print('Risk Pipeline Ready!')"]