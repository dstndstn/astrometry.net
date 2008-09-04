#! /bin/bash

Nthreads=4
Log="/home/gmaps/test/portal.log"
Cmd="/home/gmaps/test/astrometry/net/portal/watcher-script-ice.py %s 2>>$Log"

cd /home/gmaps/test/job-queue
rm queue

umask 007

export LD_LIBRARY_PATH=/home/gmaps/test/astrometry/util

/home/gmaps/test/astrometry/net/execs/watcher -D -n $Nthreads -c "$Cmd"

