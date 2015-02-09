#!/usr/bin/env bash

make clean
make astrometry_grouped_big.pdf DRAW=dot
make astrometry_grouped.pdf DRAW=circo
make astrometry_layered.pdf DRAW=dot CLUSTER=cfood-cluster_regexp GRAPH_FLAGS="--shape box"
