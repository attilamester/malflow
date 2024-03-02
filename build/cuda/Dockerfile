FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS base

WORKDIR /usr/workdir

RUN apt update && apt install -y wget && wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /miniconda && \
    eval "$(/miniconda/bin/conda shell.bash hook)" && \
    conda init
COPY environment.yml .
RUN --mount=type=cache,target=/miniconda/pkgs eval "$(/miniconda/bin/conda shell.bash hook)" && \
    CONDA_ENV_NAME=$(head -1 ./environment.yml | cut -d' ' -f2) && \
    conda env create --file=environment.yml && \
    conda activate $CONDA_ENV_NAME && \
    conda clean --all --yes && \
    conda-pack -n $CONDA_ENV_NAME -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    /venv/bin/conda-unpack && \
    find /venv -type d -name "__pycache__" -exec rm -rf {} + && \
    find /venv -follow -type f -name "*.a" -delete && \
    find /venv -follow -type f -name "*.pyc" -delete

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS main

WORKDIR /usr/workdir

COPY --from=BASE /venv /venv
RUN echo "source /venv/bin/activate" > ~/.bashrc
