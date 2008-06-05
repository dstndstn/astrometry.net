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
#include "fitsbin.h"
#include "errors.h"

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

bl* get_chunks(startree_t* s, il* wordsizes) {
    bl* chunks = bl_new(4, sizeof(fitsbin_chunk_t));
    fitsbin_chunk_t chunk;
    kdtree_t* kd = s->tree;

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "sweep";
    chunk.itemsize = sizeof(uint8_t);
    chunk.nrows = kd->ndata;
    chunk.data = s->sweep;
    chunk.userdata = &(s->sweep);
    chunk.required = FALSE;
    bl_append(chunks, &chunk);
    if (wordsizes)
        il_append(wordsizes, sizeof(uint8_t));

    fitsbin_chunk_reset(&chunk);
    chunk.tablename = "sigma_radec";
    chunk.itemsize = 2 * sizeof(float);
    chunk.nrows = kd->ndata;
    chunk.data = s->sigma_radec;
    chunk.userdata = &(s->sigma_radec);
    chunk.required = FALSE;
    bl_append(chunks, &chunk);
    if (wordsizes)
        il_append(wordsizes, sizeof(float));

    fitsbin_chunk_reset(&chunk);
    chunk.tablename = "proper_motion";
    chunk.itemsize = 2 * sizeof(float);
    chunk.nrows = kd->ndata;
    chunk.data = s->proper_motion;
    chunk.userdata = &(s->proper_motion);
    chunk.required = FALSE;
    bl_append(chunks, &chunk);
    if (wordsizes)
        il_append(wordsizes, sizeof(float));

    fitsbin_chunk_reset(&chunk);
    chunk.tablename = "sigma_pm";
    chunk.itemsize = 2 * sizeof(float);
    chunk.nrows = kd->ndata;
    chunk.data = s->sigma_pm;
    chunk.userdata = &(s->sigma_pm);
    chunk.required = FALSE;
    bl_append(chunks, &chunk);
    if (wordsizes)
        il_append(wordsizes, sizeof(float));

    fitsbin_chunk_reset(&chunk);
    chunk.tablename = "starid";
    chunk.itemsize = sizeof(uint64_t);
    chunk.nrows = kd->ndata;
    chunk.data = s->starids;
    chunk.userdata = &(s->starids);
    chunk.required = FALSE;
    bl_append(chunks, &chunk);
    if (wordsizes)
        il_append(wordsizes, sizeof(uint64_t));

    fitsbin_chunk_clean(&chunk);
    return chunks;
}

startree_t* startree_open(char* fn) {
	startree_t* s;
    bl* chunks;
    int i;
    kdtree_fits_t* io;
    char* treename = STARTREE_NAME;

	s = startree_alloc();
	if (!s)
		return s;

    io = kdtree_fits_open(fn);
	if (!io) {
        ERROR("Failed to open FITS file \"%s\"", fn);
        goto bailout;
    }

    if (!kdtree_fits_contains_tree(io, treename))
        treename = NULL;

    s->tree = kdtree_fits_read_tree(io, treename, &s->header);
    if (!s->tree) {
        ERROR("Failed to read kdtree from file \"%s\"", fn);
        goto bailout;
    }

    chunks = get_chunks(s, NULL);
    for (i=0; i<bl_size(chunks); i++) {
        fitsbin_chunk_t* chunk = bl_access(chunks, i);
        void** dest = chunk->userdata;
        kdtree_fits_read_chunk(io, chunk);
        *dest = chunk->data;
    }
    bl_free(chunks);

	return s;

 bailout:
    kdtree_fits_io_close(io);
    startree_close(s);
	return NULL;
}

/*
 void startree_close_starids(startree_t* s) {
 }

 void startree_close_motions(startree_t* s) {
 }
 */

uint64_t startree_get_starid(startree_t* s, int ind) {
    if (!s->starids)
        return 0;
    return s->starids[ind];
}

int startree_close(startree_t* s) {
	if (!s) return 0;
	if (s->inverse_perm)
		free(s->inverse_perm);
 	if (s->header)
		qfits_header_destroy(s->header);
    if (s->tree) {
        if (s->writing)
            kdtree_free(s->tree);
        else
            kdtree_fits_close(s->tree);
    }
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
    s->writing = TRUE;
	return s;
}

static int write_to_file(startree_t* s, char* fn, bool flipped) {
    bl* chunks;
    il* wordsizes = NULL;
    int i;
    kdtree_fits_t* io;
    io = kdtree_fits_open_for_writing(fn);
    if (!io) {
        ERROR("Failed to open file \"%s\" for writing kdtree", fn);
        return -1;
    }
    if (flipped) {
        if (kdtree_fits_write_tree_flipped(io, s->tree, s->header)) {
            ERROR("Failed to write (flipped) kdtree to file \"%s\"", fn);
            return -1;
        }
    } else {
        if (kdtree_fits_write_tree(io, s->tree, s->header)) {
            ERROR("Failed to write kdtree to file \"%s\"", fn);
            return -1;
        }
    }

    if (flipped)
        wordsizes = il_new(4);

    chunks = get_chunks(s, wordsizes);
    for (i=0; i<bl_size(chunks); i++) {
        fitsbin_chunk_t* chunk = bl_access(chunks, i);
        if (!chunk->data)
            continue;
        if (flipped)
            kdtree_fits_write_chunk_flipped(io, chunk, il_get(wordsizes, i));
        else
            kdtree_fits_write_chunk(io, chunk);
    }
    bl_free(chunks);

    if (flipped)
        il_free(wordsizes);
    
    kdtree_fits_io_close(io);
    return 0;
}


int startree_write_to_file(startree_t* s, char* fn) {
    return write_to_file(s, fn, FALSE);
}

int startree_write_to_file_flipped(startree_t* s, char* fn) {
    return write_to_file(s, fn, TRUE);
}

