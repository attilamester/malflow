FROM docker.io/attilamester/cuda-11.8.0.base-pytorch

WORKDIR /usr/malflow

RUN mkdir -p core/processors/bagnet && apt update && apt install -y wget && \
    wget -O core/processors/bagnet/main.py https://raw.githubusercontent.com/attilamester/pytorch-examples/feature/83/imagenet/main.py

COPY . .
