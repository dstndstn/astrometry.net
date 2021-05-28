/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef KDTREE_INTERNAL_H
#define KDTREE_INTERNAL_H

#include "astrometry/kdtree.h"

#define GLUE3(base, x, y, z) base ## _ ## x ## y ## z
#define KDMANGLE(func, e, d, t) GLUE3(func, e, d, t)

#define KD_DECLARE(func, rtn, args) \
rtn KDMANGLE(func, d, d, d)args; \
rtn KDMANGLE(func, f, f, f)args; \
rtn KDMANGLE(func, l, l, l)args; \
rtn KDMANGLE(func, d, d, u)args; \
rtn KDMANGLE(func, d, u, u)args; \
rtn KDMANGLE(func, d, d, s)args; \
rtn KDMANGLE(func, d, s, s)args

#define KD_DISPATCH(func, tt, rtn, args) \
	switch (tt) { \
	case KDTT_DOUBLE: rtn KDMANGLE(func, d, d, d)args; break; \
	case KDTT_FLOAT:  rtn KDMANGLE(func, f, f, f)args; break; \
	case KDTT_U64:    rtn KDMANGLE(func, l, l, l)args; break; \
	case KDTT_DUU: \
                      rtn KDMANGLE(func, d, u, u)args; break; \
	case KDTT_DSS: \
                      rtn KDMANGLE(func, d, s, s)args; break; \
	case KDTT_DOUBLE_U32: \
					  rtn KDMANGLE(func, d, d, u)args; break; \
	case KDTT_DOUBLE_U16: \
					  rtn KDMANGLE(func, d, d, s)args; break; \
	default: \
		fprintf(stderr, #func ": unimplemented treetype %#x.\n", tt); \
	}

/* Compute how many levels should be used if you have "N" points and you
   want "Nleaf" points in the leaf nodes.
*/
int kdtree_compute_levels(int N, int Nleaf);

#endif
