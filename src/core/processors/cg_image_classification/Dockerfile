FROM docker.io/attilamester/cuda-11.8.0.base-pytorch

WORKDIR /usr/malflow/src

RUN apt update && apt install -y wget
COPY core/processors/cg_image_classification/download_dependencies.sh .
RUN ./download_dependencies.sh main.tmp.py
COPY . .
RUN cp main.tmp.py core/processors/cg_image_classification/main.py
