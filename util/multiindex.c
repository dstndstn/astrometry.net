/*
 This file is part of the Astrometry.net suite.
 Copyright 2010 Dustin Lang.

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

#include "multiindex.h"
#include "index.h"
#include "log.h"
#include "errors.h"

// How many indices?
int multiindex_n(const multiindex_t* mi) {
	return pl_size(mi->inds);
}
// Get an index
index_t* multiindex_get(const multiindex_t* mi, int i) {
	return pl_get(mi->inds, i);
}

multiindex_t* multiindex_new(const char* skdtfn) {
	multiindex_t* mi = calloc(1, sizeof(multiindex_t));
	logverb("Reading star KD tree from %s...\n", skdtfn);
	mi->starkd = startree_open(skdtfn);
	if (!mi->starkd) {
		ERROR("Failed to open star kd-tree \"%s\"", skdtfn);
		goto bailout;
	}
	mi->inds = pl_new(16);
	return mi;
bailout:
	multiindex_free(mi);
	return NULL;
}

int multiindex_add_index(multiindex_t* mi, const char* fn) {
	anqfits_t* fits = anqfits_open(fn);
	quadfile* quads = NULL;
	codetree* codes = NULL;
	index_t* ind = NULL;
	logverb("Reading index file \"%s\"...\n", fn);
	if (!fits) {
		ERROR("Failed to open FITS file \"%s\"", fn);
		goto bailout;
	}
	logverb("Reading quads from file \"%s\"...\n", fn);
	quads = quadfile_open_fits(fits);
	if (!quads) {
		ERROR("Failed to read quads from file \"%s\"", fn);
		anqfits_close(fits);
		goto bailout;
	}
	logverb("Reading codes from file \"%s\"...\n", fn);
	codes = codetree_open_fits(fits);
	if (!codes) {
		ERROR("Failed to read quads from file \"%s\"", fn);
		quadfile_close(quads);
		anqfits_close(fits);
		goto bailout;
	}
	
	ind = index_build_from(codes, quads, mi->starkd);
	ind->fits = fits;
	if (!ind->indexname)
		ind->indexname = strdup(fn);

	pl_append(mi->inds, ind);
	return 0;
 bailout:
	if (quads) {
		quadfile_close(quads);
	}
	if (codes) {
		codetree_close(codes);
	}
	if (fits) {
		anqfits_close(fits);
	}
	return -1;
}


multiindex_t* multiindex_open(const char* skdtfn, const sl* indfns) {
	multiindex_t* mi = multiindex_new(skdtfn);
	if (!mi)
		return NULL;
	int i;
	for (i=0; i<sl_size(indfns); i++) {
		const char* fn = sl_get_const(indfns, i);
		if (multiindex_add_index(mi, fn)) {
			goto bailout;
		}
	}
	return mi;
 bailout:
	multiindex_free(mi);
	return NULL;
}

void multiindex_close(multiindex_t* mi) {
	if (!mi)
		return;
	if (mi->starkd) {
		startree_close(mi->starkd);
		mi->starkd = NULL;
	}
	if (mi->inds) {
		int i;
		for (i=0; i<pl_size(mi->inds); i++) {
			index_t* ind = pl_get(mi->inds, i);
			ind->starkd = NULL;
			index_free(ind);
		}
		pl_free(mi->inds);
		mi->inds = NULL;
	}
}

void multiindex_free(multiindex_t* mi) {
	multiindex_close(mi);
	free(mi);
}

