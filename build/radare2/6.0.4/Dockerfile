# =================
FROM python:3.10-alpine as base
# =================

WORKDIR /usr/radare2

RUN apk update && apk add curl wget graphviz graphviz-dev git make g++ build-base graphviz-libs pkgconfig musl-dev && \
    python3 -m pip install pip==25.3

RUN curl -Ls https://github.com/radareorg/radare2/releases/download/6.0.4/radare2-6.0.4.tar.xz | tar xJv && \
    cd radare2-6.0.4 && ./sys/install.sh

COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt

# =================
FROM python:3.10-alpine as main
# =================

WORKDIR /usr/radare2

COPY --from=base /usr/local/bin/python3.10 /usr/local/bin/python3.10
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/lib/libcgraph* /usr/lib/
COPY --from=base /usr/lib/libcdt* /usr/lib/
COPY --from=base /usr/lib/libgvc* /usr/lib/
COPY --from=base /usr/lib/libexpat* /usr/lib/
COPY --from=base /usr/lib/libxdot* /usr/lib/
COPY --from=base /usr/lib/libpathplan** /usr/lib/
COPY --from=base /usr/include/graphviz /usr/include/graphviz
COPY --from=base /usr/share/graphviz /usr/share/graphviz
COPY --from=base /usr/local/lib/radare2 /usr/local/lib/radare2
COPY --from=base /usr/local/bin/radare2 /usr/local/bin/radare2
COPY --from=base /usr/local/share/radare2 /usr/local/share/radare2
COPY --from=base /usr/local/bin/r2 /usr/local/bin/r2
COPY --from=base /usr/local/lib/libr_*.so /usr/local/lib/
COPY --from=base /usr/radare2/radare2-6.0.4/libr /usr/radare2/radare2-6.0.4/libr
