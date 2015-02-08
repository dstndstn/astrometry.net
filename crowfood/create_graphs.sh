#!/usr/bin/env bash

make clean
make astrometry_grouped_big.pdf DRAW=dot
make astrometry_grouped.pdf DRAW=circo
