#! /bin/bash

# This script gets run on cluster24 via ssh from oven's ssh-master.py

# It makes several Ice calls to backend servers, aggregates the results and sends them back.

# Don't write things to stdout - we pipe it to "tar" on the other end.

ICE_SOLVER=/u/dstn/go/dsolver/astrometry/net/ice/IceSolver.py

# Read jobid
read -s jobid

# Replace "-" by "/" in the jobid.
jobid=`echo "$jobid" | sed s+-+/+g`

# Create our job directory.
cd /u/dstn/go/dsolver/jobs/
mkdir -p $jobid
cd $jobid

read -s nbytes
echo "Reading $nbytes bytes of input file..." 1>&2
dd bs=1 count=$nbytes of=job.axy

# stderr goes back over the ssh tunnel...
$ICE_SOLVER $jobid job.axy > backend.stdout

# Send back all the files we generated!
tar cf - --exclude=job.axy *
