FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt install -y apt-utils && \
    apt install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    git \
    file \
    pkg-config \
    wget \
    curl \
    swig \
    netpbm \
    wcslib-dev \
    wcslib-tools \
    zlib1g-dev \
    libbz2-dev \
    libcairo2-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    libgsl-dev \
    libjpeg-dev \
    libnetpbm10-dev \
    libpng-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    python3-setuptools \
    python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pip installs
RUN for x in \
    cython \
    intel-numpy \
    intel-scipy \
    fitsio \
    matplotlib \
    astropy \
    photutils \
    zmq \
    ; do pip3 install --no-cache-dir $x; done
# psycopg2 \

RUN mkdir /src
WORKDIR /src

# Python related stuff
RUN echo "../site-packages" > /usr/local/lib/python3.6/dist-packages/site-packages.pth

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make py \
    && make extra \
    && make install INSTALL_DIR=/usr/local \
    && make clean

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHONPATH=/usr/local/lib/python

