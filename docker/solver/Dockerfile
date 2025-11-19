FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt install -y apt-utils && \
    apt install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    git \
    file \
    patch \
    gdb \
    less \
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
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Astropy needs numpy >= 1.20
#    python3-numpy \
#    python3-scipy \
#    python3-matplotlib \

# Pip installs
RUN for x in \
    numpy \
    scipy \
    matplotlib \
    fitsio \
    astropy \
    ; do pip3 install --no-cache-dir $x; done

RUN ln -s /usr/include /usr/local/include/netpbm

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHONPATH=/usr/local/lib/python

# If you wanted to use the "systemctl" scripts rather than the "run.sh" script,
# You could create and use a "nova" user.
# RUN useradd --home-dir /src --create-home --password xxx nova
# USER nova

RUN mkdir /src
WORKDIR /src

# For releases:
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry
# For local use / debugging -- copy the current directory into the container.
#ADD --chown=nova:nova . /src/astrometry

RUN git config --global --add safe.directory /src/astrometry
RUN cd astrometry \
    && make -j \
    && make py -j \
    && make extra -j
#USER root
RUN cd astrometry \
    && git config --global --add safe.directory /src/astrometry \
    && make install INSTALL_DIR=/usr/local
#USER nova
