# =================
FROM docker.io/attilamester/r2-5.8.8
# =================

WORKDIR /usr/callgraph

COPY ./requirements.txt .
RUN python3 -m pip install pip==23.0 && \
   python3 -m pip install -r requirements.txt

COPY ./src ./src
COPY ./test ./test

CMD ["python3", "-m", "pytest", "-ra", "./test/test.py"]
