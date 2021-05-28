/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include "kdtree.h"
#include "kdtree_internal_common.h"

#include "kdint_etype_l.h"
#include "kdint_dtype_l.h"
#include "kdint_ttype_l.h"

#define POINT_ED(kd, d, r, func)    (r)
#define POINT_DT(kd, d, r, func)    (r)
#define POINT_ET(kd, d, r, func)    (r)

#define POINT_TD(kd, d, r)          (r)
#define POINT_DE(kd, d, r)          (r)
#define POINT_TE(kd, d, r)          (r)

#define DIST_ED(kd, dist, func)     (dist)
#define DIST_DT(kd, dist, func)     (dist)
#define DIST_ET(kd, dist, func)     (dist)

#define DIST_TD(kd, dist)           (dist)
#define DIST_DE(kd, dist)           (dist)
#define DIST_TE(kd, dist)           (dist)

#define DIST2_ED(kd, dist2, func)   (dist2)
#define DIST2_DT(kd, dist2, func)   (dist2)
#define DIST2_ET(kd, dist2, func)   (dist2)

#define DIST2_TD(kd, dist2)         (dist2)
#define DIST2_DE(kd, dist2)         (dist2)
#define DIST2_TE(kd, dist2)         (dist2)

#define EQUAL_ED 1
#define EQUAL_DT 1
#define EQUAL_ET 1

#include "kdtree_internal.c"
#include "kdtree_internal_fits.c"

