#! /bin/bash

Nthreads=20
Log="/home/gmaps/test/portal.log"
Cmd="/home/gmaps/test/astrometry/net/portal/watcher_script_ice.py %s 2>>$Log"

cd /home/gmaps/test/job-queue
rm queue

umask 007

export LD_LIBRARY_PATH=/home/gmaps/test/astrometry/util:/home/dstn/software/ice-3.3.0/lib:/home/dstn/software/mcpp-2.7.1/lib
export PYTHONPATH=${PYTHONPATH}:/home/gmaps/test/astrometry/net/ice

/home/gmaps/test/astrometry/net/execs/watcher -D -n $Nthreads -c "$Cmd"

