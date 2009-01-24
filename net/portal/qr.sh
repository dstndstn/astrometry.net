#! /bin/bash

DIR=/home/gmaps/test/astrometry/net/portal

daemon --name=qr --pidfile=$DIR/qr.pid \
    --output=qr.log --errlog=qrdaemon.log \
    --umask=002 \
    --chdir $DIR \
    -v $* \
    python qrunner.py
