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

#include "codekd.h"
#include "kdtree_fits_io.h"
#include "starutil.h"
#include "errors.h"

static codetree* codetree_alloc() {
	codetree* s = calloc(1, sizeof(codetree));
	if (!s) {
		fprintf(stderr, "Failed to allocate a code kdtree struct.\n");
		return NULL;
	}
	return s;
}

int codetree_append_to(codetree* s, FILE* fid) {
	return kdtree_fits_append_tree_to(s->tree, s->header, fid);
}

int codetree_N(codetree* s) {
	return s->tree->ndata;
}

int codetree_nodes(codetree* s) {
	return s->tree->nnodes;
}

int codetree_D(codetree* s) {
	return s->tree->ndim;
}

qfits_header* codetree_header(codetree* s) {
	return s->header;
}

int codetree_get_permuted(codetree* s, int index) {
	if (s->tree->perm) return s->tree->perm[index];
	else return index;
}

static codetree* my_open(const char* fn, anqfits_t* fits) {
	codetree* s;
    kdtree_fits_t* io;
    char* treename = CODETREE_NAME;

	s = codetree_alloc();
	if (!s)
		return s;

	if (fits) {
		io = kdtree_fits_open_fits(fits);
		fn = fits->filename;
	} else
		io = kdtree_fits_open(fn);
	if (!io) {
        ERROR("Failed to open FITS file \"%s\"", fn);
        goto bailout;
    }
    if (!kdtree_fits_contains_tree(io, treename))
        treename = NULL;
    s->tree = kdtree_fits_read_tree(io, treename, &s->header);
	if (!s->tree) {
		ERROR("Failed to read code kdtree from file %s\n", fn);
		goto bailout;
	}

	return s;
 bailout:
	free(s);
	return NULL;
}

codetree* codetree_open_fits(anqfits_t* fits) {
	return my_open(NULL, fits);
}

codetree* codetree_open(const char* fn) {
	return my_open(fn, NULL);
}

int codetree_close(codetree* s) {
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

static int Ndata(codetree* s) {
	return s->tree->ndata;
}

void codetree_compute_inverse_perm(codetree* s) {
	// compute inverse permutation vector.
	s->inverse_perm = malloc(Ndata(s) * sizeof(int));
	if (!s->inverse_perm) {
		fprintf(stderr, "Failed to allocate code kdtree inverse permutation vector.\n");
		return;
	}
	kdtree_inverse_permutation(s->tree, s->inverse_perm);
}

int codetree_get(codetree* s, unsigned int codeid, double* code) {
	if (s->tree->perm && !s->inverse_perm) {
		codetree_compute_inverse_perm(s);
		if (!s->inverse_perm)
			return -1;
	}
	if (codeid >= Ndata(s)) {
		fprintf(stderr, "Invalid code ID: %u >= %u.\n", codeid, Ndata(s));
		return -1;
	}
	if (s->inverse_perm)
		kdtree_copy_data_double(s->tree, s->inverse_perm[codeid], 1, code);
	else
		kdtree_copy_data_double(s->tree, codeid, 1, code);
	return 0;
}

codetree* codetree_new() {
	codetree* s = codetree_alloc();
	s->header = qfits_header_default();
	if (!s->header) {
		fprintf(stderr, "Failed to create a qfits header for code kdtree.\n");
		free(s);
		return NULL;
	}
	qfits_header_add(s->header, "AN_FILE", AN_FILETYPE_CODETREE, "This file is a code kdtree.", NULL);
	return s;
}

int codetree_write_to_file(codetree* s, const char* fn) {
	return kdtree_fits_write(s->tree, fn, s->header);
}

int codetree_write_to_file_flipped(codetree* s, const char* fn) {
	return kdtree_fits_write_flipped(s->tree, fn, s->header);
}
