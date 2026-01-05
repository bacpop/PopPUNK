FROM python:3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                          cmake \
                          gfortran \
                          libarmadillo-dev \
                          libeigen3-dev \
                          libopenblas-dev \
                          python3-graph-tool

RUN git clone https://github.com/somme89/rapidNJ.git && \
    cd rapidNJ && \
    make && \
    mv ./bin/rapidnj /usr/bin &&\
    cd .. && rm -r rapidNJ



RUN pip install pybind11[global]

COPY . /src
WORKDIR /src
RUN pip install .

RUN pip install cmake plotly ffmpeg

RUN git clone https://github.com/bacpop/mandrake.git && \
    cd mandrake && \
    python setup.py install && \
    cd .. && rm -r mandrake
