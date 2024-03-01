FROM docker.io/attilamester/cuda-11.8.0.base-pytorch

WORKDIR /usr/malflow

RUN mkdir -p src/core/processors/bagnet && apt update && apt install -y wget && \
    wget -O src/core/processors/bagnet/main.py https://raw.githubusercontent.com/attilamester/pytorch-examples/feature/83/imagenet/main.py

COPY . .

ENTRYPOINT ["/bin/bash", "-c", "source /venv/bin/activate && cd src && python3 -m core.processors.bagnet.main --use-module-definitions train_definitions.py"]
