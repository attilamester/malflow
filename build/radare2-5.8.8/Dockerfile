# =================
FROM python:3.8-alpine as base
# =================

WORKDIR /usr/callgraph

RUN apk update && apk add curl wget graphviz graphviz-dev git make g++ build-base graphviz-libs pkgconfig musl-dev && \
    python3 -m pip install pip==23.0

RUN curl -Ls https://github.com/radareorg/radare2/releases/download/5.8.8/radare2-5.8.8.tar.xz | tar xJv && \
    cd radare2-5.8.8 && sed -i "265 i patch-capstone capstone-patch:" ./shlr/Makefile && sed -i 's/CS_VER=4.*/CS_VER=5\.0.1/' ./shlr/Makefile &&  ./sys/install.sh

COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt

# =================
FROM python:3.8-alpine as main
# =================

WORKDIR /usr/callgraph

COPY --from=base /usr/local/bin/python3.8 /usr/local/bin/python3.8
COPY --from=base /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=base /usr/lib/libcgraph* /usr/lib/
COPY --from=base /usr/lib/libcdt* /usr/lib/
COPY --from=base /usr/lib/libgvc* /usr/lib/
COPY --from=base /usr/lib/libxdot* /usr/lib/
COPY --from=base /usr/lib/libpathplan** /usr/lib/
COPY --from=base /usr/include/graphviz /usr/include/graphviz
COPY --from=base /usr/share/graphviz /usr/share/graphviz
COPY --from=base /usr/local/lib/radare2 /usr/local/lib/radare2
COPY --from=base /usr/local/bin/radare2 /usr/local/bin/radare2
COPY --from=base /usr/local/share/radare2 /usr/local/share/radare2
COPY --from=base /usr/local/bin/r2 /usr/local/bin/r2
COPY --from=base /usr/local/lib/libr_*.so /usr/local/lib/
COPY --from=base /usr/callgraph/radare2-5.8.8/libr /usr/callgraph/radare2-5.8.8/libr
