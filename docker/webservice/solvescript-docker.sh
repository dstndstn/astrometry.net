#! /bin/bash

# Run via:
#   python process_submissions.py --solve-locally=PATH_TO/solvescript-docker.sh

set -e

jobid=$1
axyfile=$2

BACKEND="/usr/local/bin/astrometry-engine"
CFG="/index/docker.cfg"
export TMP=/tmp

$BACKEND -v -c $CFG $axyfile -j $jobid

