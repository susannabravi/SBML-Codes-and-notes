# Base image with CUDA 12.4 and Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt /app/

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install flash-attn (requires CUDA compiler present in devel image)
RUN pip install --no-cache-dir flash_attn --no-build-isolation

# Uncomment if you want to auto-run something
# CMD ["python", "download_models.py"]
