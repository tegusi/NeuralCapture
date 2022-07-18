FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake tmux vim ffmpeg libsm6 libxext6 && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY env.yaml /tmp/environment.yaml
RUN conda env update -n base -f /tmp/environment.yaml && \
    conda clean -aqy && \
    rm -rf /opt/miniconda/pkgs && \
    find / -type d -name __pycache__ -prune -exec rm -rf {} \; && \
    rm /tmp/environment.yaml