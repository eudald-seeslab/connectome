# Use the official NVIDIA CUDA image as a base
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio

# Install torch-geometric
RUN pip3 install torch-geometric

# Install torch-scatter, torch-sparse, and other PyTorch Geometric dependencies
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Set the working directory
WORKDIR /workspace

# Define the entrypoint
CMD ["bash"]
