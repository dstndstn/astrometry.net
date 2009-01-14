#! /bin/bash

DIR=/home/gmaps/test/astrometry/net/portal

daemon --name=qr --pidfile=$DIR/qr.pid --umask=002 --output=qr.log --errlog=qrdaemon.log \
    --chdir $DIR \
    $* \
    python qrunner.py
