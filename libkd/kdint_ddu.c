/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include "kdtree.h"
#include "kdtree_internal_common.h"

#include "kdint_etype_d.h"
#include "kdint_dtype_d.h"
#include "kdint_ttype_u.h"

#define POINT_ED(kd, d, r, func)    (r)
#define POINT_DT(kd, d, r, func)    (func(POINT_SCALE(kd, d, r)))
#define POINT_ET(kd, d, r, func)    (func(POINT_SCALE(kd, d, r)))

#define POINT_TD(kd, d, r)          POINT_INVSCALE(kd, d, r)
#define POINT_DE(kd, d, r)          (r)
#define POINT_TE(kd, d, r)          POINT_INVSCALE(kd, d, r)

#define DIST_ED(kd, dist, func)     (dist)
#define DIST_DT(kd, dist, func)     (func(DIST_SCALE(kd, dist)))
#define DIST_ET(kd, dist, func)     (func(DIST_SCALE(kd, dist)))

#define DIST_TD(kd, dist)           DIST_INVSCALE(kd, dist)
#define DIST_DE(kd, dist)           (dist)
#define DIST_TE(kd, dist)           DIST_INVSCALE(kd, dist)

#define DIST2_ED(kd, dist2, func)   (dist2)
#define DIST2_DT(kd, dist2, func)   (func(DIST2_SCALE(kd, dist2)))
#define DIST2_ET(kd, dist2, func)   (func(DIST2_SCALE(kd, dist2)))

#define DIST2_TD(kd, dist2)         DIST2_INVSCALE(kd, dist2)
#define DIST2_DE(kd, dist2)         (dist2)
#define DIST2_TE(kd, dist2)         DIST2_INVSCALE(kd, dist2)

#define EQUAL_ED 1
#define EQUAL_DT 0
#define EQUAL_ET 0

#include "kdtree_internal.c"
#include "kdtree_internal_fits.c"

