FROM ubuntu:22.04

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py312_24.5.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && mkdir /opt/conda/_conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

ARG REPO_BRANCH="main"

WORKDIR /repo

# git clone and conda create
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/milzj/MCRNNOC
RUN cd MCRNNOC && conda env create -f environment.yml
RUN cd && conda init bash && exec bash
