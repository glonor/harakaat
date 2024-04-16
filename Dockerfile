# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV CONDA_ENV_NAME=har

# Basic setup
RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists


# Set working directory
WORKDIR /workspace/project


# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"


# Switch to bash shell
SHELL ["/bin/bash", "-c"]

# Create a new conda environment
COPY environment.yaml ./
RUN conda env create -f environment.yaml -n ${CONDA_ENV_NAME} \
    && conda clean --all --yes \
    && rm environment.yaml


# Set ${CONDA_ENV_NAME} to default virtual environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
