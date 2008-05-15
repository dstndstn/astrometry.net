/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "starkd.h"
#include "kdtree_fits_io.h"
#include "starutil.h"

static startree_t* startree_alloc() {
	startree_t* s = calloc(1, sizeof(startree_t));
	if (!s) {
		fprintf(stderr, "Failed to allocate a star kdtree struct.\n");
		return NULL;
	}
	return s;
}

int startree_N(startree_t* s) {
	return s->tree->ndata;
}

int startree_nodes(startree_t* s) {
	return s->tree->nnodes;
}

int startree_D(startree_t* s) {
	return s->tree->ndim;
}

qfits_header* startree_header(startree_t* s) {
	return s->header;
}

static void sweep_tablesize(kdtree_t* kd, extra_table* tab) {
	tab->nitems = kd->ndata;
}

startree_t* startree_open(char* fn) {
	startree_t* s;
	extra_table extras[1];
	extra_table* sweep = extras;

	memset(extras, 0, sizeof(extras));

	sweep->name = "sweep";
	sweep->datasize = sizeof(uint8_t);
	sweep->nitems = 0;
	sweep->required = 0;
	sweep->compute_tablesize = sweep_tablesize;

	s = startree_alloc();
	if (!s)
		return s;

	s->tree = kdtree_fits_read_extras(fn, NULL, &s->header, extras,
									  sizeof(extras)/sizeof(extra_table));
	if (!s->tree) {
		fprintf(stderr, "Failed to read star kdtree from file %s\n", fn);
		goto bailout;
	}

	s->sweep = sweep->ptr;

	return s;

 bailout:
	free(s);
	return NULL;
}

int startree_close(startree_t* s) {
	if (!s) return 0;
	if (s->inverse_perm)
		free(s->inverse_perm);
 	if (s->header)
		qfits_header_destroy(s->header);
	if (s->tree)
		kdtree_fits_close(s->tree);
	free(s);
	return 0;
}

static int Ndata(startree_t* s) {
	return s->tree->ndata;
}

void startree_compute_inverse_perm(startree_t* s) {
	// compute inverse permutation vector.
	s->inverse_perm = malloc(Ndata(s) * sizeof(int));
	if (!s->inverse_perm) {
		fprintf(stderr, "Failed to allocate star kdtree inverse permutation vector.\n");
		return;
	}
	kdtree_inverse_permutation(s->tree, s->inverse_perm);
}

int startree_get(startree_t* s, int starid, double* posn) {
	if (s->tree->perm && !s->inverse_perm) {
		startree_compute_inverse_perm(s);
		if (!s->inverse_perm)
			return -1;
	}
	if (starid >= Ndata(s)) {
		fprintf(stderr, "Invalid star ID: %u >= %u.\n", starid, Ndata(s));
                assert(0);
		return -1;
	}
	if (s->inverse_perm) {
		kdtree_copy_data_double(s->tree, s->inverse_perm[starid], 1, posn);
	} else {
		kdtree_copy_data_double(s->tree, starid, 1, posn);
	}
	return 0;
}

startree_t* startree_new() {
	startree_t* s = startree_alloc();
	s->header = qfits_header_default();
	if (!s->header) {
		fprintf(stderr, "Failed to create a qfits header for star kdtree.\n");
		free(s);
		return NULL;
	}
	qfits_header_add(s->header, "AN_FILE", AN_FILETYPE_STARTREE, "This file is a star kdtree.", NULL);
	return s;
}

int startree_write_to_file(startree_t* s, char* fn) {
	if (s->sweep) {
		extra_table extras[1];
		extra_table* sweep = extras;
		memset(extras, 0, sizeof(extras));
		sweep->name = "sweep";
		sweep->datasize = sizeof(uint8_t);
		sweep->nitems = s->tree->ndata;
		sweep->ptr = s->sweep;
		sweep->found = 1;
		return kdtree_fits_write_extras(s->tree, fn, s->header, extras,
										sizeof(extras)/sizeof(extra_table));
	} else
		return kdtree_fits_write(s->tree, fn, s->header);
}
