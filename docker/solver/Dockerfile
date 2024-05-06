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
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pip installs
RUN for x in \
    fitsio \
    astropy \
    ; do pip3 install --no-cache-dir $x; done

RUN ln -s /usr/include /usr/local/include/netpbm

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHONPATH=/usr/local/lib/python

# For compatibility with the service scripts
# (e.g. https://github.com/dstndstn/astrometry.net/blob/main/net/nova-jobs.service),
# we use a user nova here.  Change the default password for better
# security here.
RUN useradd -m nova -p x
USER nova
WORKDIR /home/nova

# Since this docker file is part of the repo, copy the current directory
# (assumed to be the astrometry.net repo base directory) into the container.

ADD . /home/nova/astrometry
RUN cd astrometry \
    && make -j \
    && make py -j \
    && make extra -j \
    && make install INSTALL_DIR=/usr/local
