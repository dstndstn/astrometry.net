#! /bin/bash

# Run via:
#   python process_submissions.py --solve-locally=$(pwd)/solvescript.sh

set -e

jobid=$1
axyfile=$2

BACKEND="/home/nova/astrometry/solver/astrometry-engine"
CFG="/home/nova/astrometry/net/nova.cfg"
export TMP=/tmp

$BACKEND -v -c $CFG $axyfile -j $jobid

