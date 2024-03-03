FROM python:3.8-alpine as base

RUN apk update && apk add make automake gcc g++ subversion python3-dev && \
    python3 -m pip install pip==23.0 && python3 -m pip install tensorboard==2.14.0 six==1.16.0

CMD python3 -m tensorboard.main --logdir /log_dir --port 6006 --host 0.0.0.0
