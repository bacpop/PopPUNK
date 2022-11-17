FROM python:3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                          cmake \
                          gfortran \
                          libarmadillo-dev \
                          libeigen3-dev \
                          libopenblas-dev \
                          software-properties-common

RUN git clone https://github.com/somme89/rapidNJ.git && \
    cd rapidNJ && \
    make && \
    mv ./bin/rapidnj /usr/bin &&\
    cd .. && rm -r rapidNJ

# Special snowflake treatment for graph_tool because the maintainer
# refuses to make the package pip-installable, meaning we can't depend
# on it in a sane way. Least pain seems to be to install via apt, even
# though that might result in slightly incorrect versions (built
# around system python).
#
# The alternative would be to build it from source
# https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#manual-compilation
# but it takes over an hour to compile apparently, so that sounds not
# ideal.
RUN add-apt-repository https://downloads.skewed.de/apt && \
    apt-key adv --keyserver keyserver.ubuntu.com \
            --recv-key 612DEFB798507F25 && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3-graph-tool && \
    ln -s /usr/lib/python3/dist-packages/graph_tool \
      /usr/local/lib/python3.10/site-packages

RUN pip install pybind11[global]

COPY . /src
WORKDIR /src
RUN pip install .

RUN pip install cmake plotly ffmpeg

RUN git clone https://github.com/bacpop/mandrake.git && \
    cd mandrake && \
    python setup.py install && \
    cd .. && rm -r mandrake
