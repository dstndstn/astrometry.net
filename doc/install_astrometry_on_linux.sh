#!/bin/sh
# TODO: a few of the commands require ‘yes/no’ answers, how to avoid this?

WORKSPACE_DIR="/tmp"

# basic linux install/build libs
cd $WORKSPACE_DIR && \
apt install make && \
apt-get update && \
apt-get install build-essential && \

# python and other support
apt-get install python python-pip netpbm zlib1g-dev libcairo2-dev libjpeg-dev && \
pip install numpy scipy pyfits && \

# cfitsio
cd $WORKSPACE_DIR && \
rm -f cfitsio_latest.tar.gz* && \
wget https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz && \
tar xvzf cfitsio_latest.tar.gz && \
cd cfitsio && \
./configure && \
make && \
make install && \

# astrometry
cd $WORKSPACE_DIR && \
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
wget -r -nd -P /usr/local/astrometry/data/ "data.astrometry.net/4200/"

