# Base image with CUDA 12.4 and Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, and basic utilities
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    unzip \
    build-essential \
    ninja-build \
    libxt6 libxrender1 libxext6 libsm6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Quarto CLI
RUN wget https://quarto.org/download/latest/quarto-linux-amd64.deb && \
    dpkg -i quarto-linux-amd64.deb && \
    rm quarto-linux-amd64.deb

# Set working directory
WORKDIR /app

# Install PyTorch FIRST with exact CUDA 12.4 compatibility
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install transformers and related packages
RUN pip install transformers==4.54.0 peft==0.7.1 accelerate==1.9.0

# Copy requirements
COPY requirements.txt /app/

# Install other requirements (excluding flash-attn to avoid conflicts)
RUN pip install packaging jupyter nbformat
RUN grep -v "flash" requirements.txt > requirements_no_flash.txt || true
RUN pip install --no-cache-dir -r requirements_no_flash.txt || pip install --no-cache-dir -r requirements.txt

# Install Flash Attention with specific compatibility for PyTorch 2.6.0 + CUDA 12.4
# This will either find a compatible wheel or compile from source
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Set fallback environment variable in case flash attention still has issues
ENV DISABLE_FLASH_ATTN=0

# Default command
CMD ["/bin/bash"]