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

#ifndef STAR_KD_H
#define STAR_KD_H

#include <stdint.h>
#include "kdtree.h"
#include "qfits.h"

#define AN_FILETYPE_STARTREE "SKDT"

struct startree {
	kdtree_t* tree;
	qfits_header* header;
	int* inverse_perm;
	uint8_t* sweep;
};
typedef struct startree startree;


startree* startree_open(char* fn);

int startree_N(startree* s);

int startree_nodes(startree* s);

int startree_D(startree* s);

qfits_header* startree_header(startree* s);

int startree_get(startree* s, unsigned int starid, double* posn);

int startree_close(startree* s);

void startree_compute_inverse_perm(startree* s);

// for writing
startree* startree_new();

int startree_write_to_file(startree* s, char* fn);

#endif
