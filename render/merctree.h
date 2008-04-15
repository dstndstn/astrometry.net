/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef MERCTREE_H
#define MERCTREE_H

#include "kdtree.h"
#include "qfits.h"

struct merc_flux {
	float rflux;
	float bflux;
	float nflux;
};
typedef struct merc_flux merc_flux;

struct merc_cached_stats {
	merc_flux flux;
};
typedef struct merc_cached_stats merc_stats;

struct merctree {
	kdtree_t* tree;
	merc_stats* stats;
	merc_flux* flux;
	qfits_header* header;
	int* inverse_perm;
};
typedef struct merctree merctree;


merctree* merctree_open(char* fn);

//int merctree_get(merctree* s, uint mercid, double* posn);

int merctree_close(merctree* s);

//void merctree_compute_inverse_perm(merctree* s);

void merctree_compute_stats(merctree* m);

// for writing
merctree* merctree_new();

int merctree_write_to_file(merctree* s, char* fn);

#endif
