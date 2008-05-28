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
#include "errors.h"
#include "ioutils.h"

bool index_has_ids(index_t* index) {
    return index->use_ids;
}

void get_filenames(const char* indexname,
                   char** quadfn,
                   char** ckdtfn,
                   char** skdtfn,
                   bool* singlefile) {
    char* basename;
    if (file_readable(indexname)) {
        // assume single-file index.
        *ckdtfn = strdup(indexname);
        *skdtfn = strdup(indexname);
        *quadfn = strdup(indexname);
        *singlefile = TRUE;
        return;
    }
    if (ends_with(indexname, ".quad.fits")) {
        basename = strdup(indexname);
        basename[strlen(indexname)-10] = '\0';
    } else
        basename = strdup(indexname);
    *ckdtfn = mk_ctreefn(basename);
    *skdtfn = mk_streefn(basename);
    *quadfn = mk_quadfn (basename);
    *singlefile = FALSE;
    free(basename);
    return;
}

bool index_is_file_index(const char* filename) {
    char* ckdtfn, *skdtfn, *quadfn;
    bool singlefile;
    index_t* ind;
    bool rtn;

    get_filenames(filename, &quadfn, &ckdtfn, &skdtfn, &singlefile);
    if (!(file_readable(quadfn) &&
          (singlefile || (file_readable(ckdtfn) && file_readable(skdtfn))))) {
        rtn = FALSE;
        goto finish;
    }
    if (!(qfits_is_fits(quadfn) &&
          (singlefile || (qfits_is_fits(ckdtfn) && qfits_is_fits(skdtfn))))) {
        rtn = FALSE;
        goto finish;
    }

    ind = index_load(filename, INDEX_ONLY_LOAD_METADATA);
    rtn = (ind != NULL);

 finish:
    free(ckdtfn);
    free(skdtfn);
    free(quadfn);
    return rtn;
}

int index_get_scale_and_id(const char* filename,
                           double* scalelo, double* scalehi,
                           int* indexid, int* healpix, int* hpnside) {
    index_t* ind = index_load(filename, 0);
    if (!ind) {
        return -1;
    }
    if (scalelo)
        *scalelo = quadfile_get_index_scale_lower_arcsec(ind->quads);
    if (scalehi)
        *scalehi = quadfile_get_index_scale_upper_arcsec(ind->quads);
    if (indexid)
        *indexid = ind->indexid;
    if (healpix)
        *healpix = ind->healpix;
    if (hpnside)
        *hpnside = ind->hpnside;
    index_close(ind);
    return 0;
}

index_t* index_load(const char* indexname, int flags) {
	char *codetreefname=NULL, *quadfname=NULL, *startreefname=NULL;
	index_t* index = calloc(1, sizeof(index_t));
	index->indexname = indexname;
    bool singlefile;

	if (flags & INDEX_ONLY_LOAD_METADATA)
		logverb("Loading metadata for %s...\n", indexname);

    get_filenames(indexname, &quadfname, &codetreefname, &startreefname, &singlefile);

	// Read .skdt file...
	logverb("Reading star KD tree from %s...\n", startreefname);
	index->starkd = startree_open(startreefname);
	if (!index->starkd) {
		ERROR("Failed to read star kdtree from file %s", startreefname);
        goto bailout;
	}
	free(startreefname);
    startreefname = NULL;
	//logverb("  (%d stars, %d nodes).\n", startree_N(index->starkd), startree_nodes(index->starkd));

	index->index_jitter = qfits_header_getdouble(index->starkd->header, "JITTER", DEFAULT_INDEX_JITTER);
	//logverb("Setting index jitter to %g arcsec.\n", index->index_jitter);

	if (flags & INDEX_ONLY_LOAD_SKDT)
		return index;

	// Read .quad file...
	logverb("Reading quads file %s...\n", quadfname);
	index->quads = quadfile_open(quadfname);
	if (!index->quads) {
        ERROR("Failed to read quads file from %s", quadfname);
        goto bailout;
	}
	free(quadfname);
    quadfname = NULL;
	index->index_scale_upper = quadfile_get_index_scale_upper_arcsec(index->quads);
	index->index_scale_lower = quadfile_get_index_scale_lower_arcsec(index->quads);
	index->indexid = index->quads->indexid;
	index->healpix = index->quads->healpix;
	index->hpnside = index->quads->hpnside;

	//logverb("  (%d stars: %i, Quads: %i.\n", index->quads->numstars, index->quads->numquads);

	logverb("Index scale: [%g, %g] arcmin, [%g, %g] arcsec\n",
            index->index_scale_lower / 60.0, index->index_scale_upper / 60.0,
            index->index_scale_lower, index->index_scale_upper);

	// Read .ckdt file...
	logverb("Reading code KD tree from %s...\n", codetreefname);
	index->codekd = codetree_open(codetreefname);
	if (!index->codekd) {
		ERROR("Failed to read code kdtree from file %s", codetreefname);
        goto bailout;
    }
	free(codetreefname);
    codetreefname = NULL;

	// check for CIRCLE field in ckdt header...
	index->circle = qfits_header_getboolean(index->codekd->header, "CIRCLE", 0);
	if (!index->circle) {
		ERROR("Code kdtree does not contain the CIRCLE header.");
        goto bailout;
	}

	//logverb("  (%d quads, %d nodes).\n", codetree_N(index->codekd), codetree_nodes(index->codekd));

	// New indexes are cooked such that cx < dx for all codes, but not
	// all of the old ones are like this.
    index->cx_less_than_dx = qfits_header_getboolean(index->codekd->header, "CXDX", FALSE);

	if (flags & INDEX_USE_IDS)
        index->use_ids = TRUE;

	if (flags & INDEX_ONLY_LOAD_METADATA)
		index_close(index);

	return index;

 bailout:
    free(startreefname);
    free(quadfname);
    free(codetreefname);
    index_close(index);
    return NULL;
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
