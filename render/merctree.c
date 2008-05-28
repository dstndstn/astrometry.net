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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "merctree.h"
#include "kdtree_fits_io.h"
#include "starutil.h"

static merctree* merctree_alloc() {
	merctree* s = calloc(1, sizeof(merctree));
	if (!s) {
		fprintf(stderr, "Failed to allocate a merctree struct.\n");
		return NULL;
	}
	return s;
}

void get_stats_chunk(merctree* mt, fitsbin_chunk_t* chunk) {
    fitsbin_chunk_init(chunk);
    chunk->tablename = "merc_stats";
    chunk->itemsize = sizeof(merc_stats);
    chunk->nrows = mt->tree->nnodes;
    chunk->data = mt->stats;
    //chunk->userdata = &(mt->stats);
    chunk->required = TRUE;
}
void get_flux_chunk(merctree* mt, fitsbin_chunk_t* chunk) {
    fitsbin_chunk_init(chunk);
    chunk->tablename = "merc_flux";
    chunk->itemsize = sizeof(merc_flux);
    chunk->nrows = mt->tree->ndata;
    chunk->data = mt->flux;
    //chunk->userdata = &(mt->flux);
    chunk->required = TRUE;
}

merctree* merctree_open(char* fn) {
	merctree* s;
    kdtree_fits_t* io;
    fitsbin_chunk_t chunk;
	
	s = merctree_alloc();
	if (!s)
		return s;
    io = kdtree_fits_open(fn);
    if (!io) {
		fprintf(stderr, "Failed to open file %s\n", fn);
        goto bailout;
        return NULL;
    }
	s->tree = kdtree_fits_read_tree(io, NULL, &s->header);
	if (!s->tree) {
		fprintf(stderr, "Failed to read merc kdtree from file %s\n", fn);
		goto bailout;
	}

    get_stats_chunk(s, &chunk);
    if (kdtree_fits_read_chunk(io, &chunk)) {
		fprintf(stderr, "Failed to read merc stats from file %s\n", fn);
		goto bailout;
    }
	s->stats = chunk.data;

    get_flux_chunk(s, &chunk);
    if (kdtree_fits_read_chunk(io, &chunk)) {
		fprintf(stderr, "Failed to read merc flux from file %s\n", fn);
		goto bailout;
    }
	s->flux = chunk.data;

	return s;

 bailout:
    
    merctree_close(s);
	return NULL;
}

int merctree_close(merctree* s) {
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

void merctree_compute_stats(merctree* m) {
	int i;
	kdtree_t* kd = m->tree;
	if (kd->perm) {
		fprintf(stderr, "Error: merctree_compute_stats requires a permuted tree.\n");
		assert(0);
		return;
	}
	for (i=kd->nnodes-1; i>=0; i--) {
		merc_flux* stats = &(m->stats[i].flux);
		if (KD_IS_LEAF(m->tree, i)) {
			int j;
			int L, R;
			stats->rflux = stats->bflux = stats->nflux = 0.0;
			L = kdtree_left(m->tree, i);
			R = kdtree_right(m->tree, i);
			for (j=L; j<=R; j++) {
				stats->rflux += m->flux[j].rflux;
				stats->bflux += m->flux[j].bflux;
				stats->nflux += m->flux[j].nflux;
			}
		} else {
			merc_flux *flux1, *flux2;
			flux1 = &(m->stats[KD_CHILD_LEFT(i) ].flux);
			flux2 = &(m->stats[KD_CHILD_RIGHT(i)].flux);
			stats->rflux = flux1->rflux + flux2->rflux;
			stats->bflux = flux1->bflux + flux2->bflux;
			stats->nflux = flux1->nflux + flux2->nflux;
		}
	}
}

merctree* merctree_new() {
	merctree* s = merctree_alloc();
	s->header = qfits_header_default();
	if (!s->header) {
		fprintf(stderr, "Failed to create a qfits header for merc kdtree.\n");
		free(s);
		return NULL;
	}
	qfits_header_add(s->header, "AN_FILE", "MKDT", "This file is a merc kdtree.", NULL);
	return s;
}

int merctree_write_to_file(merctree* s, char* fn) {
    kdtree_fits_t* io;
    fitsbin_chunk_t chunk;

    io = kdtree_fits_open_for_writing(fn);
    if (!io) {
        fprintf(stderr, "Failed to open file %s for writing.\n", fn);
        return -1;
    }
    if (kdtree_fits_write_tree(io, s->tree, s->header)) {
        fprintf(stderr, "Failed to write merc kdtree to file %s.\n", fn);
        return -1;
    }

    get_stats_chunk(s, &chunk);
    if (kdtree_fits_write_chunk(io, &chunk)) {
        fprintf(stderr, "Failed to write merc stats to file %s.\n", fn);
        return -1;
    }
    fitsbin_chunk_clean(&chunk);

    get_flux_chunk(s, &chunk);
    if (kdtree_fits_write_chunk(io, &chunk)) {
        fprintf(stderr, "Failed to write merc flux to file %s.\n", fn);
        return -1;
    }
    fitsbin_chunk_clean(&chunk);

    if (kdtree_fits_io_close(io)) {
        fprintf(stderr, "Failed to close output file %s.\n", fn);
        return -1;
    }

    return 0;
}

