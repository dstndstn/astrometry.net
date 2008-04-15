#! /bin/bash

# This script gets run on clusterX via ssh from oven.

# Careful about writing things to stdout - we pipe it to tar on the other end.

QUADS=/u/dstn/amd-an-2/quads

BACKEND="${QUADS}/backend"
BACKEND_CFG="${QUADS}/backend-test.cfg"
AXY="job.axy"

# Read jobid
read -s jobid

# Replace "-" by "/" in the jobid.
jobid=`echo "$jobid" | sed s+-+/+g`

# Create our job directory.
cd /nobackup/dstn/web-data
mkdir -p $jobid
cd $jobid

# Read input files from stdin
tar xf -

# stderr goes back over the ssh tunnel to the log file on oven.
$BACKEND -c $BACKEND_CFG -i blind.in -v $AXY > backend.out

# Send back all the files we generated!
tar cf - *
