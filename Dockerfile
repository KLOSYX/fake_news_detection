# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.9    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV LANG C.UTF-8
ADD requirements.txt /tmp/

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
&& chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
# ENV HOME=/root
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Download anaconda install
RUN curl -sLo miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    # Turn executable attribute
    chmod +x miniconda3.sh && \
    # Install Anaconda
    bash miniconda3.sh -b && \
    # Remove Installation file
    rm miniconda3.sh

# Add Anaconda to the Path
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda3/bin:$PATH

# Init conda
RUN cd $HOME/miniconda3/bin && sudo ./conda init --all --system

# Run bash. From there you can use pip or conda to install needed packages
CMD ["bash"]

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    GIT_CLONE="git clone --depth 10" && \
    sudo apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------    
    sudo DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
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
        fish \
        htop \
        proxychains4

# ==================================================================
# install requirements
# ------------------------------------------------------------------
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
        $PIP_INSTALL -r /tmp/requirements.txt
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN sudo ldconfig && \
        sudo apt-get clean && \
        sudo apt-get autoremove && \
        sudo rm -rf /var/lib/apt/lists/* /tmp/*
