#! /bin/bash

cd /home/gmaps/test/astrometry/net/server
LOG=worker-$(hostname)-$PPID
echo "Logging to $LOG.{out,err}"
touch $LOG.out $LOG.err
python ./solver.py /home/gmaps/INDEXES/500 >> $LOG.out 2>> $LOG.err

