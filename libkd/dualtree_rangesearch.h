/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef DUALTREE_RANGE_SEARCH_H
#define DUALTREE_RANGE_SEARCH_H

extern double RANGESEARCH_NO_LIMIT;

#include "kdtree.h"

// note, 'xind' and 'yind' are indices IN THE KDTREE; to get back to
// 'normal' ordering you must use the kdtree permutation vector.
typedef void (*result_callback)(void* extra, int xind, int yind,
								double dist2);

typedef void (*progress_callback)(void* extra, int ydone);

typedef double (*dist2_function)(void* px, void* py, int D);

void dualtree_rangesearch(kdtree_t* xtree, kdtree_t* ytree,
						  double mindist, double maxdist,
						  anbool notself,
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
