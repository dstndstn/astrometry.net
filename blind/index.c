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

#include "index.h"
#include "fileutil.h"
#include "log.h"

bool index_has_ids(index_t* index) {
    return index->use_ids;
}

index_t* index_load(const char* indexname, int flags) {
	char *treefname, *quadfname, *startreefname;

	index_t* index = calloc(1, sizeof(index_t));
	index->indexname = indexname;

	if (flags & INDEX_ONLY_LOAD_METADATA)
		logverb("Loading only metadata for %s...\n", indexname);

	// Read .skdt file...
	startreefname = mk_streefn(indexname);
	logverb("Reading star KD tree from %s...\n", startreefname);
	index->starkd = startree_open(startreefname);
	if (!index->starkd) {
		logerr("Failed to read star kdtree %s\n", startreefname);
		free_fn(startreefname);
		free(index);
		return NULL;
	}
	free_fn(startreefname);
	logverb("  (%d stars, %d nodes).\n", startree_N(index->starkd), startree_nodes(index->starkd));

	index->index_jitter = qfits_header_getdouble(index->starkd->header, "JITTER", DEFAULT_INDEX_JITTER);
	logverb("Setting index jitter to %g arcsec.\n", index->index_jitter);

	if (flags & INDEX_ONLY_LOAD_SKDT)
		return index;

	// Read .quad file...
	quadfname = mk_quadfn(indexname);
	logverb("Reading quads file %s...\n", quadfname);
	index->quads = quadfile_open(quadfname);
	if (!index->quads) {
		logerr("Couldn't read quads file %s\n", quadfname);
		free_fn(quadfname);
		return NULL;
	}
	free_fn(quadfname);
	index->index_scale_upper = quadfile_get_index_scale_upper_arcsec(index->quads);
	index->index_scale_lower = quadfile_get_index_scale_lower_arcsec(index->quads);
	index->indexid = index->quads->indexid;
	index->healpix = index->quads->healpix;
	index->hpnside = index->quads->hpnside;

	logverb("Stars: %i, Quads: %i.\n", index->quads->numstars, index->quads->numquads);

	logverb("Index scale: [%g, %g] arcmin, [%g, %g] arcsec\n",
	       index->index_scale_lower / 60.0, index->index_scale_upper / 60.0,
	       index->index_scale_lower, index->index_scale_upper);

	// Read .ckdt file...
	treefname = mk_ctreefn(indexname);
	logverb("Reading code KD tree from %s...\n", treefname);
	index->codekd = codetree_open(treefname);
	if (!index->codekd) {
		logerr("Failed to read code kdtree %s\n", treefname);
		free_fn(treefname);
		return NULL;
	}
	free_fn(treefname);

	// check for CIRCLE field in ckdt header...
	index->circle = qfits_header_getboolean(index->codekd->header, "CIRCLE", 0);
	if (!index->circle) {
		logerr("Code kdtree does not contain the CIRCLE header.\n");
		return NULL;
	}

	logverb("  (%d quads, %d nodes).\n", codetree_N(index->codekd), codetree_nodes(index->codekd));

	// New indexes are cooked such that cx < dx for all codes, but not
	// all of the old ones are like this.
	if (qfits_header_getboolean(index->codekd->header, "CXDX", 0))
		index->cx_less_than_dx = TRUE;
	else
		index->cx_less_than_dx = FALSE;

	if (flags & INDEX_USE_IDS)
        index->use_ids = TRUE;

	if (flags & INDEX_ONLY_LOAD_METADATA)
		index_close(index);

	return index;
}

void index_close(index_t* index)
{
	if (!index) return;
	if (index->starkd)
		startree_close(index->starkd);
	if (index->codekd)
		codetree_close(index->codekd);
	if (index->quads)
		quadfile_close(index->quads);
	index->starkd = NULL;
	index->codekd = NULL;
	index->quads = NULL;
	free(index);
}
