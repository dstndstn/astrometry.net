#! /bin/bash

# This script gets run on clusterX via ssh from oven's ssh-master.py

# Don't write things to stdout - we pipe it to "tar" on the other end.

BACKEND=/u/dstn/go/dsolver/astrometry/blind/backend
BACKEND_CONFIG=/u/dstn/go/dsolver/backend-conf/backend-`hostname`.cfg

# Read jobid
read -s jobid

# Replace "-" by "/" in the jobid.
jobid=`echo "$jobid" | sed s+-+/+g`

# Create our job directory.
cd /u/dstn/go/dsolver/jobs/
mkdir -p $jobid
cd $jobid
mkdir `hostname`
cd `hostname`
# Read tarred input data...
#tar xf -

read -s nbytes
echo "Will read $nbytes bytes..." > /dev/stderr
dd bs=1 count=$nbytes of=job.axy

# stderr goes back over the ssh tunnel...
$BACKEND -c $BACKEND_CONFIG -E -v job.axy -C ../cancel > backend.stdout

if [ -e solved ]; then
    touch ../cancel
fi

# Send back all the files we generated!
tar cf - --exclude=job.axy *
