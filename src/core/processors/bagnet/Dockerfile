# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /usr/malcallgraph

RUN apt update && apt install -y wget && wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash /tmp/anaconda.sh -b -p /anaconda && \
    eval "$(/anaconda/bin/conda shell.bash hook)" && \
    conda init && \
    conda update -n base -c defaults conda
COPY ./ ./
RUN eval "$(/anaconda/bin/conda shell.bash hook)" && \
    conda env create --name malcallgraph-train --file=environment.yml && \
    conda activate malcallgraph-train
