/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2009 Dustin Lang, Keir Mierle and Sam Roweis.

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
#ifndef QUAD_UTILS_H
#define QUAD_UTILS_H

#include "starkd.h"
#include "codefile.h"
#include "quadfile.h"

int quad_compute_code(const unsigned int* quad, int dimquads, startree_t* starkd, 
					  double* code);

void quad_enforce_invariants(unsigned int* quad, double* code,
							 int dimquads, int dimcodes);

bool quad_obeys_invariants(unsigned int* quad, double* code,
						   int dimquads, int dimcodes);

/**
 Warning: this may permute "quad"!
 */
void quad_write(codefile* codes, quadfile* quads,
				unsigned int* quad, startree_t* starkd,
				int dimquads, int dimcodes);

void quad_write_const(codefile* codes, quadfile* quads,
					  const unsigned int* quad, startree_t* starkd,
					  int dimquads, int dimcodes);

#endif
