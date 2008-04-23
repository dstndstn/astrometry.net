#! /bin/bash

cd /nobackup30/dstn/an/astrometry/net/server
LOG=worker-$(hostname)-$PPID
echo "Logging to $LOG.{out,err}"
touch $LOG.out $LOG.err
python ./solver.py /nobackup/dstn/solver-indexes >> $LOG.out 2>> $LOG.err

