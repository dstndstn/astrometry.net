#! /bin/bash

# This script gets run on neuron-0-X via ssh from oven's ssh-master.py

# Don't write things to stdout - we pipe it to "tar" on the other end.

#BACKEND="/data1/dstn/astrometry/blind/backend"
#BACKEND_CONFIG=/data1/dstn/dsolver/backend-config/backend-`hostname -s`.cfg
BACKEND_CLIENT="python /data1/dstn/dsolver/astrometry/blind/backend_client.py"

JOBDIR=/data1/dstn/dsolver/jobs/


# Read jobid
read -s jobid

# Replace "-" by "/" in the jobid.
jobid=`echo "$jobid" | sed s+-+/+g`

# Create our job directory.
cd $JOBDIR
mkdir -p $jobid
cd $jobid
mkdir `hostname`
cd `hostname`
# Read tarred input data...
#tar xf -

read -s nbytes
echo "Will read $nbytes bytes..." 1>&2
dd bs=1 count=$nbytes of=job.axy

# stderr goes back over the ssh tunnel...
#$BACKEND -c $BACKEND_CONFIG -E -v job.axy -C ../cancel > backend.stdout

$BACKEND_CLIENT `pwd`/job.axy `pwd`/../cancel > backend.stdout

if [ -e solved ]; then
    touch ../cancel
fi

# Send back all the files we generated!
tar cf - --exclude=job.axy *
