# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.9    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04
ENV LANG C.UTF-8
# Copy requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://mirrors.ustc.edu.cn/pypi/web/simple" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        openssh-server \
        screen \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        scikit-image \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        && \
# ==================================================================
# jupyter
# ------------------------------------------------------------------
    $PIP_INSTALL \
        jupyter \
        && \
# ==================================================================
# requirmemnts
# ------------------------------------------------------------------
    $PIP_INSTALL -r /tmp/requirements.txt && \
    $PIP_INSTALL deepspeed && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
