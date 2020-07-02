# Use NVIDIA pre-built docker container
FROM nvidia/cuda:10.2-base-ubuntu16.04

# Set the working directory to /app
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip

# Install python packages
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    SimpleITK \
    scikit-image