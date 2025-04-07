# Install base Nvidia image with Ubuntu 22.04.
# Change CUDA wrt your version.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install base python packages.
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get -y install python3-pip git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Remember to replace CUDA version to your CUDA version.

# Create workspace directory.
RUN mkdir workspace
WORKDIR /workspace

# Get files from git.
RUN git clone https://github.com/susannabravi/SBML-Codes-and-notes.git
WORKDIR /workspace/SBML-Codes-and-notes

COPY test.py /workspace/SBML-Codes-and-notes

# Install required packages.
RUN pip3 install -r /workspace/SBML-Codes-and-notes/requirements.txt

# EXPOSE 8503   # Not for the moment.

CMD ["python3", "-m", "display_database"]