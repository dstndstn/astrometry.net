/*
  This file is part of libkd.
  Copyright 2006-2008 Dustin Lang and Keir Mierle.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "kdtree.h"
#include "kdtree_internal_common.h"

#include "kdint_etype_f.h"
#include "kdint_dtype_f.h"
#include "kdint_ttype_f.h"

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

