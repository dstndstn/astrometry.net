#!/bin/sh

WORKSPACE_DIR="/tmp"

# basic linux install/build libs, python, and other support
cd $WORKSPACE_DIR && \
apt install -y make && \
apt-get update && \
apt-get install -y build-essential python python-pip netpbm zlib1g-dev libcairo2-dev libjpeg-dev libcfitsio-dev && \
pip install numpy scipy pyfits && \

# astrometry
rm -f astrometry.net-latest.tar.gz* && \
wget http://astrometry.net/downloads/astrometry.net-latest.tar.gz && \
tar xvzf astrometry.net-latest.tar.gz && \
cd astrometry.net-* && \
make CFITS_INC="-I$WORKSPACE_DIR/cfitsio/include" CFITS_LIB="-L$WORKSPACE_DIR/cfitsio/lib -lcfitsio" && \
make py && \
make extra && \
make CFITS_INC="-I$WORKSPACE_DIR/cfitsio/include" CFITS_LIB="-L$WORKSPACE_DIR/cfitsio/lib -lcfitsio" install && \

# download and install index files
rm -rf /usr/local/astrometry/data/* && \
wget -r -nd -np -P /usr/local/astrometry/data/ "data.astrometry.net/4100/" && \
wget -r -nd -np -P /usr/local/astrometry/data/ "data.astrometry.net/5000/"
