FROM ubuntu:22.04

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN mkdir -p miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
RUN bash miniconda3/miniconda.sh -b -u -p miniconda3
RUN rm -rf miniconda3/miniconda.sh
RUN echo "export PATH=miniconda3/bin/conda:$PATH" > /etc/profile.d/conda.sh
ENV PATH miniconda3/bin/conda:$PATH

RUN miniconda3/bin/conda init bash
RUN miniconda3/bin/conda init zsh

ARG REPO_BRANCH="main"

WORKDIR /repo

# git clone and conda create
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/milzj/MCRNNOC
RUN cd MCRNNOC && miniconda3/bin/conda env create -f environment.yml
RUN cd && miniconda3/bin/conda init bash && exec bash
