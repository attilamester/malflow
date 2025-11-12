# =================
FROM docker.io/attilamester/radare2:6.0.4
# =================

WORKDIR /usr/malflow

COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY ./src ./src
COPY ./test ./test
