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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "kdtree.h"
#include "kdtree_internal.h"

KD_DECLARE(kdtree_build, kdtree_t*, (kdtree_t* kd, void* data, int N, int D, int Nleaf, unsigned int options));

/* Build a tree from an array of data, of size N*D*sizeof(real) */
/* If the root node is level 0, then maxlevel is the level at which there may
 * not be enough points to keep the tree complete (i.e. last level) */
kdtree_t* KDFUNC(kdtree_build)
	 (kdtree_t* kd, void *data, int N, int D, int Nleaf,
	  int treetype, unsigned int options) {

	KD_DISPATCH(kdtree_build, treetype, kd=, (kd, data, N, D, Nleaf, options))

	if (kd) {
		kd->treetype = treetype;
	}
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


