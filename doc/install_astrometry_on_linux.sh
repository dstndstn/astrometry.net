#!/bin/sh

WORKSPACE_DIR="/tmp"

# basic linux install/build libs, python, and other support
cd $WORKSPACE_DIR && \
apt update && \
apt install -y make build-essential python3 python3-pip netpbm libnetpbm10-dev zlib1g-dev libcairo2-dev libjpeg-dev libcfitsio-dev libbz2-dev wget wcslib-dev swig && \
pip3 install numpy scipy fitsio && \

# astrometry
rm -f astrometry.net-latest.tar.gz* && \
wget http://astrometry.net/downloads/astrometry.net-latest.tar.gz && \
tar xvzf astrometry.net-latest.tar.gz && \
cd astrometry.net-* && \
make && \
make py && \
make extra && \
make install && \

# download and install index files
rm -rf /usr/local/astrometry/data/* && \
wget -r -nd -np -P /usr/local/astrometry/data/ "data.astrometry.net/4100/" && \
wget -r -nd -np -P /usr/local/astrometry/data/ "data.astrometry.net/5000/"
