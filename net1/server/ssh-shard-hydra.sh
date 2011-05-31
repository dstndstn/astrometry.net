#! /bin/bash

# This script gets run on neuron-0-X via ssh from oven's ssh-master.py

# Don't write things to stdout - we pipe it to "tar" on the other end.

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
mkdir `hostname -s`
cd `hostname -s`

# Read input file length and contents.
read -s nbytes
echo "Will read $nbytes bytes..." 1>&2
dd bs=1 count=$nbytes of=job.axy

# stderr goes back over the ssh tunnel...

$BACKEND_CLIENT `pwd`/job.axy `pwd`/../cancel > backend.stdout &

while [ 1 ]; do
    # Wait for a command from the master, with 1-second timeout...
    read -t 1 command
    if [ x$command != x ]; then
	echo "Got command: $command" 1>&2
	#echo "Killing job..."
	#kill %%
	echo "Cancelling job..." 1>&2
	touch `pwd`/../cancel
	echo "Waiting..." 1>&2
	wait
	break;
    fi
    # Check if the process finished.
    jobs %% > /dev/null 2>/dev/null
    jobstat=$?
    if [ $jobstat -ne 0 ]; then
	echo "Job finished." 1>&2
	break;
    fi
done

# Send back all the files we generated
tar cf - --ignore-failed-read --exclude=job.axy * ../solved
