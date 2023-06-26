FROM python:3.10

RUN apt-get update

# NB These steps follow the same order as the listed requirements for manual compilation detailed here:
# https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#manual-compilation

# Install C++ compiler
RUN apt-get install -y --no-install-recommends build-essential

# Install Boost dev tools
# TODO: Try replacing libboost-all-dev with just the ones required, as included in debian dockerfile
RUN apt-get install -y --no-install-recommends g++ python3-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev

RUN apt-get install -y --no-install-recommends libexpat1-dev

RUN apt-get install -y --no-install-recommends python3-scipy

RUN pip3 install numpy

RUN apt-get install -y --no-install-recommends libcgal-dev

RUN apt-get install -y --no-install-recommends libsparsehash-dev

# graph drawing - not including graphviz
RUN apt-get install -y --no-install-recommends \
    gir1.2-gtk-3.0 \
    libcairomm-1.0-dev \
    python3-cairo-dev \
    python3-cairo \
    python3-matplotlib

RUN pip3 install pycairo

# so we can generate configure script
RUN apt-get install -y --no-install-recommends autoconf

# get and build source, and install to default Python location
RUN git clone https://git.skewed.de/count0/graph-tool.git
WORKDIR /graph-tool
RUN ./autogen.sh
RUN ./configure
RUN make
RUN make install











