# Base image with CUDA 12.4 and Ubuntu 22.04
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

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
# COPY download_models.py /app/
# COPY scripts/ /app/
# COPY utils/ /app/

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download models
# RUN python download_models.py

#CMD ["python"]
