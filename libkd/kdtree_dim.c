/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "kdtree.h"
#include "kdtree_internal.h"

KD_DECLARE(kdtree_build_2, kdtree_t*, (kdtree_t* kd, void *data, int N, int D, int Nleaf, int treetype, unsigned int options, double* minval, double* maxval));

/* Build a tree from an array of data, of size N*D*sizeof(real) */
/* If the root node is level 0, then maxlevel is the level at which there may
 * not be enough points to keep the tree complete (i.e. last level) */
kdtree_t* KDFUNC(kdtree_build_2)
     (kdtree_t* kd, void *data, int N, int D, int Nleaf,
      int treetype, unsigned int options,
      double* minval, double* maxval) {

    KD_DISPATCH(kdtree_build_2, treetype, kd=, (kd, data, N, D, Nleaf, treetype, options, minval, maxval));
    return kd;
}

/* Range seach */
kdtree_qres_t* KDFUNC(kdtree_rangesearch)
     (const kdtree_t *kd, const void *pt, double maxd2) {
    return KDFUNC(kdtree_rangesearch_options_reuse)(kd, NULL, pt, maxd2, KD_OPTIONS_COMPUTE_DISTS | KD_OPTIONS_SORT_DISTS);
}

kdtree_qres_t* KDFUNC(kdtree_rangesearch_nosort)
     (const kdtree_t *kd, const void *pt, double maxd2) {
    return KDFUNC(kdtree_rangesearch_options_reuse)(kd, NULL, pt, maxd2, KD_OPTIONS_COMPUTE_DISTS);
}

kdtree_qres_t* KDFUNC(kdtree_rangesearch_options)
     (const kdtree_t *kd, const void *pt, double maxd2, int options) {
    return KDFUNC(kdtree_rangesearch_options_reuse)(kd, NULL, pt, maxd2, options);
}

KD_DECLARE(kdtree_rangesearch_options, kdtree_qres_t*, (const kdtree_t* kd, kdtree_qres_t* res, const void* pt, double maxd2, int options));

kdtree_qres_t* KDFUNC(kdtree_rangesearch_options_reuse)
     (const kdtree_t *kd, kdtree_qres_t* res, const void *pt, double maxd2, int options) {
    assert(kd->fun.rangesearch);
    return kd->fun.rangesearch(kd, res, pt, maxd2, options);
}


