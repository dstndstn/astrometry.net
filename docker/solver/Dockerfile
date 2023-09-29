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
    && apt-get clean

# Pip installs
RUN for x in \
    fitsio \
    astropy \
    ; do pip3 install --no-cache-dir $x; done

RUN mkdir /src
WORKDIR /src

RUN ln -s /usr/include /usr/local/include/netpbm

# Since this docker file is part of the repo, we directly map the repo folder to the container as the working directory.
# Note this is a one time mapping. Docker will pack the . folder (the repo folder) into a tar file and upload to the container.
# Due to some retrictions of Docker, we cannot directly use ../../, but have to specify the building context in the command line
# and refer it as . here.
# Note this doesn't include the index files, which are supposed to be downloaded locally and attach to the container through `docker run`.
ADD . /src/astrometry
RUN cd astrometry \
    && make -j \
    && make py -j \
    && make extra -j\
    && make install INSTALL_DIR=/usr/local

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHONPATH=/usr/local/lib/python