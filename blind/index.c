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
#include "log.h"
#include "errors.h"
#include "ioutils.h"

bool index_has_ids(index_t* index) {
    return index->use_ids;
}

int index_get_quad_dim(const index_t* index) {
    return quadfile_dimquads(index->quads);
}

int index_nquads(const index_t* index) {
    return quadfile_nquads(index->quads);
}

int index_nstars(const index_t* index) {
    return startree_N(index->starkd);
}

void get_filenames(const char* indexname,
                   char** quadfn,
                   char** ckdtfn,
                   char** skdtfn,
                   bool* singlefile) {
    char* basename;
    if (ends_with(indexname, ".quad.fits")) {
        basename = strdup(indexname);
        basename[strlen(indexname)-10] = '\0';
    } else {
        char* fits;
        if (file_readable(indexname)) {
            // assume single-file index.
            *ckdtfn = strdup(indexname);
            *skdtfn = strdup(indexname);
            *quadfn = strdup(indexname);
            *singlefile = TRUE;
            return;
        }
        asprintf(&fits, "%s.fits", indexname);
        if (file_readable(fits)) {
            // assume single-file index.
            indexname = fits;
            *ckdtfn = strdup(indexname);
            *skdtfn = strdup(indexname);
            *quadfn = strdup(indexname);
            *singlefile = TRUE;
            free(fits);
            return;
        }
        free(fits);
        basename = strdup(indexname);
    }
    asprintf(ckdtfn, "%s.ckdt.fits", basename);
    asprintf(skdtfn, "%s.skdt.fits", basename);
    asprintf(quadfn, "%s.quad.fits", basename);
    *singlefile = FALSE;
    free(basename);
    return;
}

bool index_is_file_index(const char* filename) {
    char* ckdtfn, *skdtfn, *quadfn;
    bool singlefile;
    index_t* ind;
    bool rtn = FALSE;

    get_filenames(filename, &quadfn, &ckdtfn, &skdtfn, &singlefile);
    if (!file_readable(quadfn)) {
        ERROR("Index file %s is not readable.\n", quadfn);
        goto finish;
    }
    if (!singlefile) {
        if (!file_readable(ckdtfn)) {
            ERROR("Index file %s is not readable.\n", ckdtfn);
            goto finish;
        }
        if (!file_readable(skdtfn)) {
            ERROR("Index file %s is not readable.\n", skdtfn);
            goto finish;
        }
    }

    if (!(qfits_is_fits(quadfn) &&
          (singlefile || (qfits_is_fits(ckdtfn) && qfits_is_fits(skdtfn))))) {
        if (singlefile)
            ERROR("Index file %s is not FITS.\n", quadfn);
        else
            ERROR("Index files %s , %s , and %s are not FITS.\n",
                  quadfn, skdtfn, ckdtfn);
        rtn = FALSE;
        goto finish;
    }

    ind = index_load(filename, INDEX_ONLY_LOAD_METADATA);
    rtn = (ind != NULL);
    if (!rtn) {
        if (singlefile)
            ERROR("File %s does not contain an index.\n", quadfn);
        else
            ERROR("Files %s , %s , and %s do not contain an index.\n",
                  quadfn, skdtfn, ckdtfn);
    }

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
