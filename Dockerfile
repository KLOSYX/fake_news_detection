# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV LANG C.UTF-8
# Copy requirements.txt
COPY requirements.txt ./
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
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
# ==================================================================
# python
# ------------------------------------------------------------------
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
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
    $PIP_INSTALL -r requirements.txt && \
    rm requirements.txt && \
    $PIP_INSTALL deepspeed && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
