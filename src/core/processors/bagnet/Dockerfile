FROM docker.io/attilamester/cuda-11.8.0.base-pytorch

WORKDIR /usr/malflow

RUN apt update && apt install -y wget && \
    wget -O /tmp/main.py https://raw.githubusercontent.com/attilamester/pytorch-examples/feature/83/imagenet/main.py

COPY . .
RUN cp /tmp/main.py core/processors/bagnet/main.py
