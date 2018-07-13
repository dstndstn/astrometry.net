/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef DUALTREE_RANGE_SEARCH_H
#define DUALTREE_RANGE_SEARCH_H

extern double RANGESEARCH_NO_LIMIT;

#include "astrometry/kdtree.h"

// note, 'xind' and 'yind' are indices IN THE KDTREE; to get back to
// 'normal' ordering you must use the kdtree permutation vector.
typedef void (*result_callback)(void* extra, int xind, int yind,
                                double dist2);

typedef void (*progress_callback)(void* extra, int ydone);

typedef double (*dist2_function)(void* px, void* py, int D);

void dualtree_rangesearch(kdtree_t* xtree, kdtree_t* ytree,
                          double mindist, double maxdist,
                          int notself,
                          dist2_function distsquared,
                          result_callback callback,
                          void* param,
                          progress_callback progress,
                          void* progress_param);

/*
 void dualtree_rangecount(kdtree_t* x, kdtree_t* y,
 double mindist, double maxdist,
 dist2_function distsquared,
 int* counts);
 */

#endif
