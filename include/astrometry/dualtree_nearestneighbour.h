/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef DUALTREE_NEAREST_NEIGHBOUR_H
#define DUALTREE_NEAREST_NEIGHBOUR_H

#include "astrometry/kdtree.h"

void dualtree_nearestneighbour(kdtree_t* xtree, kdtree_t* ytree, double maxdist2,
                               double** nearest_d2, int** nearest_ind,
                               int** count_within_range,
                               int notself);

#endif

