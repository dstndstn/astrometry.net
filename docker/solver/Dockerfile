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
     python3-setuptools \
    python3-wheel \
    python3-pil \
   python3-numpy \
    python3-scipy \
    python3-matplotlib \
    source-extractor \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#    python3-tk \
#    plplot-driver-cairo \
#    libplplot-dev \

#    numpy \
#    scipy \
#    matplotlib \
# Pip installs
RUN for x in \
    fitsio \
    astropy \
    ; do pip3 install --no-cache --break-system-packages $x; done

# to help astrometry.net find netpbm (yuck)
RUN ln -s /usr/include /usr/local/include/netpbm

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
# Add the directory where the astrometry.net code puts its python libraries to PYTHONPATH
ENV PYTHONPATH=/usr/local/lib/python

RUN mkdir /src
WORKDIR /src

RUN git config --global --add safe.directory /src/astrometry

# Build flag - don't optimize for the build computer's CPU
ENV ARCH_FLAGS=-march=x86-64-v3

RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
  && cd astrometry \
  && make -j \
  && make py -j \
  && make extra -j \
  && make install INSTALL_DIR=/usr/local
