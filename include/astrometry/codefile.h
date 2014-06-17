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

#ifndef CODEFILE_H_
#define CODEFILE_H_

#include <sys/types.h>
#include <stdio.h>

#include "astrometry/qfits_header.h"

#include "astrometry/starutil.h"
#include "astrometry/fitsbin.h"
#include "astrometry/quadfile.h"
#include "astrometry/starkd.h"

// util:
void codefile_compute_star_code(const double* starxyz, double* code, int dimquads);

void codefile_compute_field_code(const double* xy, double* code, int dimquads);




typedef struct {
	int numcodes;
	int numstars;

	int dimcodes;

	// upper bound
	double index_scale_upper;
	// lower bound
	double index_scale_lower;
	// unique ID of this index
	int indexid;
	// healpix covered by this index
	int healpix;
    // Nside of the healpixelization
    int hpnside;

    fitsbin_t* fb;

	// when reading:
	double* codearray;
} codefile_t;

int codefile_close(codefile_t* cf);

int codefile_dimcodes(const codefile_t* cf);

void codefile_get_code(const codefile_t* cf, int codeid, double* code);

codefile_t* codefile_open(const char* fn);

codefile_t* codefile_open_for_writing(const char* fname);

codefile_t* codefile_open_in_memory();

// when in-memory
int codefile_switch_to_reading(codefile_t* cf);

int codefile_write_header(codefile_t* cf);

int codefile_write_code(codefile_t* cf, double* code);

int codefile_fix_header(codefile_t* cf);

qfits_header* codefile_get_header(const codefile_t* cf);



void quad_write(codefile_t* codes, quadfile_t* quads,
				unsigned int* quad, startree_t* starkd,
				int dimquads, int dimcodes);

void quad_write_const(codefile_t* codes, quadfile_t* quads,
					  const unsigned int* quad, startree_t* starkd,
					  int dimquads, int dimcodes);


#endif
