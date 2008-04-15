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

#ifndef KDTREE_INTERNAL_H
#define KDTREE_INTERNAL_H

#include "kdtree.h"


#define GLUE3(base, x, y, z) base ## _ ## x ## y ## z
#define KDMANGLE(func, e, d, t) GLUE3(func, e, d, t)


#define KD_DECLARE(func, rtn, args) \
rtn KDMANGLE(func, d, d, d)args; \
rtn KDMANGLE(func, f, f, f)args; \
rtn KDMANGLE(func, d, d, u)args; \
rtn KDMANGLE(func, d, u, u)args; \
rtn KDMANGLE(func, d, d, s)args; \
rtn KDMANGLE(func, d, s, s)args


#define KD_DISPATCH(func, tt, rtn, args) \
	switch (tt) { \
	case KDTT_DOUBLE: rtn KDMANGLE(func, d, d, d)args; break; \
	case KDTT_FLOAT:  rtn KDMANGLE(func, f, f, f)args; break; \
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
