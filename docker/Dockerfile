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

# Copy and install requirements
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install packaging jupyter nbformat && \
    pip install --no-cache-dir -r requirements.txt

# Install flash-attn
RUN pip install --no-cache-dir flash_attn --no-build-isolation

# Default command
CMD ["/bin/bash"]
