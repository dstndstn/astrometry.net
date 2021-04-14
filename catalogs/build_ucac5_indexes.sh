#!/bin/sh
# Build the full set of UCAC5 indexes
# Copyright 2021 Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill
# Licensed under a 3-clause BSD style license - see LICENSE

# This script assumes that uncompressed UCAC5 files (z001..z900) are present in the current directory
# and that the following Astrometry.net binaries are on the path:
# 1. ucac5tofits
# 2. build-astrometry-index

# Modify this as appropriate; set to 0 to use the original UCAC epochs
EPOCH=2022

# Set index file overlapping to approx. half of the FOV in degrees
MARGIN=2

# Set to 1 if you want tag-along data in the indexes:
#   ID, RA_GAIA, DEC_GAIA, RA_GAIA_ERR, DEC_GAIA_ERR, FLAGS, NUM_POS, UCAC_EPOCH,
#   RA_UCAC, DEC_UCAC, PM_RA, PM_DEC, PM_RA_ERR, PM_DEC_ERR, GMAG, RMAG, JMAG, HMAG, KMAG
FULL=0

# Number of parallel index build jobs (perhaps number of CPU cores if enough RAM)
JOBS=8

# Common prefix for index files
PREFIX="ucac5-index-"


DATE=$(date "+%y%m%d")

# Store temporary files in the current dir
mkdir tmp


# Build scales from 0 to 2 with NSIDE=2 (48 healpix)
NSIDE=2

ucac5tofits -N $NSIDE -e $EPOCH -m $MARGIN -f $FULL -o ucac5_%02i.fits z???

NJOBS=0
HP=0
while [ $HP -lt 48 ]; do
  for SCALE in 0 1 2; do
    # shellcheck disable=SC2046 disable=SC2086
    build-astrometry-index -A RA -D DEC -S MAG -E -t tmp -j 0.06 -P $SCALE -H $HP -s $NSIDE \
      -I $DATE$(printf %02i $SCALE) -i ucac5_$(printf %02i $HP).fits \
      -o $PREFIX$(printf %02i $SCALE)-$(printf %02i $HP).fits &
    NJOBS=$((NJOBS+1))
    if [ $NJOBS -ge $JOBS ]; then
      NJOBS=0
      wait
    fi
  done
  HP=$((HP+1))
done
wait

rm ucac5_??.fits tmp/*


# Build scales from 3 to 6 with NSIDE=1 (12 healpix)
NSIDE=1

ucac5tofits -N $NSIDE -e $EPOCH -m $MARGIN -f $FULL -o ucac5_%02i.fits z???

NJOBS=0
HP=0
while [ $HP -lt 12 ]; do
  for SCALE in 3 4 5 6; do
    # shellcheck disable=SC2046 disable=SC2086
    build-astrometry-index -A RA -D DEC -S MAG -E -t tmp -j 0.06 -P $SCALE -H $HP -s $NSIDE \
      -I $DATE$(printf %02i $SCALE) -i ucac5_$(printf %02i $HP).fits \
      -o $PREFIX$(printf %02i $SCALE)-$(printf %02i $HP).fits &
    NJOBS=$((NJOBS+1))
    if [ $NJOBS -ge $JOBS ]; then
      NJOBS=0
      wait
    fi
  done
  HP=$((HP+1))
done
wait

rm ucac5_??.fits tmp/*


# Build scales from 7 to 19 with a single all-sky healpix

ucac5tofits -N 0 -e $EPOCH -f $FULL -o ucac5.fits z???

NJOBS=0
for SCALE in 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  # shellcheck disable=SC2046 disable=SC2086
  build-astrometry-index -A RA -D DEC -S MAG -E -t tmp -j 0.06 -P $SCALE \
    -I $DATE$(printf %02i $SCALE) -i ucac5.fits \
    -o $PREFIX$(printf %02i $SCALE).fits &
  NJOBS=$((NJOBS+1))
  if [ $NJOBS -ge $JOBS ]; then
    NJOBS=0
    wait
  fi
done
wait

# Done
rm -rf ucac5.fits tmp
