/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include "keywords.h"

static anbool DIST_FUNC_MANGLE(bb_point_mindist2_exceeds, FUNC_SUFFIX)
     (const PTYPE* lo, const PTYPE* hi, const PTYPE* point, int dim, DISTTYPE maxd2)
{
    DISTTYPE d2 = 0;
    DISTTYPE newd2;
    DISTTYPE delta;
    int i;
#if defined(KD_DIM)
    dim = KD_DIM;
#endif
    for (i = 0; i < dim; i++) {
        if (point[i] < lo[i])
            delta = lo[i] - point[i];
        else if (point[i] > hi[i])
            delta = point[i] - hi[i];
        else
            continue;
#if defined(DELTAMAX)
        if (delta > DELTAMAX)
            return TRUE;
#endif
        newd2 = d2 + delta * delta;
        if (CAN_OVERFLOW && (newd2 < d2))
            // Int overflow!
            return TRUE;
        if (newd2 > maxd2)
            return TRUE;
        d2 = newd2;
    }
    return FALSE;
}

Unused // don't warn if unused.
static void DIST_FUNC_MANGLE(bb_point_mindist2_bailout, FUNC_SUFFIX)
     (const PTYPE* lo, const PTYPE* hi, const PTYPE* point, int dim, DISTTYPE maxd2,
      anbool* bailedout, DISTTYPE* p_d2)
{
    DISTTYPE d2 = 0;
    DISTTYPE newd2;
    DISTTYPE delta;
    int i;
#if defined(KD_DIM)
    dim = KD_DIM;
#endif
    for (i = 0; i < dim; i++) {
        if (point[i] < lo[i])
            delta = lo[i] - point[i];
        else if (point[i] > hi[i])
            delta = point[i] - hi[i];
        else
            continue;
#if defined(DELTAMAX)
        if (delta > DELTAMAX) {
            *bailedout = TRUE;
            return;
        }
#endif
        newd2 = d2 + delta * delta;
        if (CAN_OVERFLOW && (newd2 < d2)) {
            // Int overflow!
            *bailedout = TRUE;
            return;
        }
        if (newd2 > maxd2) {
            *bailedout = TRUE;
            return;
        }
        d2 = newd2;
    }
    *p_d2 = d2;
}

static anbool DIST_FUNC_MANGLE(bb_point_maxdist2_exceeds, FUNC_SUFFIX)
     (const PTYPE* lo, const PTYPE* hi, const PTYPE* point, int dim, DISTTYPE maxd2) {
    DISTTYPE d2 = 0;
    DISTTYPE newd2;
    PTYPE delta1, delta2;
    DISTTYPE delta;
    int i;
#if defined(KD_DIM)
    dim = KD_DIM;
#endif
    for (i = 0; i < dim; i++) {
        delta1 = (point[i] > lo[i]) ? point[i] - lo[i] : lo[i] - point[i];
        delta2 = (point[i] > hi[i]) ? point[i] - hi[i] : hi[i] - point[i];
        delta = (delta1 > delta2 ? delta1 : delta2);
#if defined(DELTAMAX)
        if (delta > DELTAMAX)
            return TRUE;
#endif
        newd2 = d2 + delta * delta;
        if (CAN_OVERFLOW && (newd2 < d2))
            // Int overflow!
            return TRUE;
        if (newd2 > maxd2)
            return TRUE;
        d2 = newd2;
    }
    return FALSE;
}

