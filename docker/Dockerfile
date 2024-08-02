FROM ubuntu:22.04

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py312_24.5.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN echo "export PATH=/miniconda3/bin/conda:$PATH" > /etc/profile.d/conda.sh
ENV PATH /miniconda3/bin/conda:$PATH

ARG REPO_BRANCH="main"

WORKDIR /repo

# git clone and conda create
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/milzj/MCRNNOC
RUN cd MCRNNOC && conda env create -f environment.yml
RUN cd && conda init bash && exec bash
