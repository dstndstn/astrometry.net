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
#include "tic.h"

bool index_meta_overlaps_scale_range(index_meta_t* meta,
                                     double quadlo, double quadhi) {
    return !((quadlo > meta->index_scale_upper) ||
             (quadhi < meta->index_scale_lower));
}

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

// Ugh!
static void get_filenames(const char* indexname,
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
            if (ckdtfn) *ckdtfn = strdup(indexname);
            if (skdtfn) *skdtfn = strdup(indexname);
            if (quadfn) *quadfn = strdup(indexname);
            *singlefile = TRUE;
            return;
        }
        asprintf(&fits, "%s.fits", indexname);
        if (file_readable(fits)) {
            // assume single-file index.
            indexname = fits;
            if (ckdtfn) *ckdtfn = strdup(indexname);
            if (skdtfn) *skdtfn = strdup(indexname);
            if (quadfn) *quadfn = strdup(indexname);
            *singlefile = TRUE;
            free(fits);
            return;
        }
        free(fits);
        basename = strdup(indexname);
    }
    if (ckdtfn) asprintf(ckdtfn, "%s.ckdt.fits", basename);
    if (skdtfn) asprintf(skdtfn, "%s.skdt.fits", basename);
    if (quadfn) asprintf(quadfn, "%s.quad.fits", basename);
    *singlefile = FALSE;
    free(basename);
    return;
}

char* index_get_quad_filename(const char* indexname) {
    char* quadfn;
    bool singlefile;
    if (!index_is_file_index(indexname))
        return NULL;
    get_filenames(indexname, &quadfn, NULL, NULL, &singlefile);
    return quadfn;
}

bool index_is_file_index(const char* filename) {
    char* ckdtfn, *skdtfn, *quadfn;
    bool singlefile;
    //index_meta_t meta;
    bool rtn = TRUE;

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

    /* This is a bit expensive...

     if (index_get_meta(filename, &meta)) {
     if (singlefile)
     ERROR("File %s does not contain an index.\n", quadfn);
     else
     ERROR("Files %s , %s , and %s do not contain an index.\n",
     quadfn, skdtfn, ckdtfn);
     rtn = FALSE;
     }
     */

 finish:
    free(ckdtfn);
    free(skdtfn);
    free(quadfn);
    return rtn;
}

int index_get_meta(const char* filename, index_meta_t* meta) {
    index_t* ind = index_load(filename, 0);
    if (!ind)
        return -1;
    memcpy(meta, &(ind->meta), sizeof(index_meta_t));
    meta->indexname = strdup(meta->indexname);
    index_close(ind);
    return 0;
}

index_t* index_load(const char* indexname, int flags) {
    struct timeval tv1, tv2;

	char *codetreefname=NULL, *quadfname=NULL, *startreefname=NULL;
    bool singlefile;
	index_t* index = calloc(1, sizeof(index_t));
	//index->meta.indexname = strdup(indexname);

	if (flags & INDEX_ONLY_LOAD_METADATA)
		logverb("Loading metadata for %s...\n", indexname);

    gettimeofday(&tv1, NULL);
    get_filenames(indexname, &quadfname, &codetreefname, &startreefname, &singlefile);
    gettimeofday(&tv2, NULL);
    debug("get_filenames took %g ms.\n", millis_between(&tv1, &tv2));

	index->meta.indexname = strdup(quadfname);

	// Read .skdt file...
	logverb("Reading star KD tree from %s...\n", startreefname);
    gettimeofday(&tv1, NULL);
	index->starkd = startree_open(startreefname);
    gettimeofday(&tv2, NULL);
    debug("reading skdt took %g ms.\n", millis_between(&tv1, &tv2));
	if (!index->starkd) {
		ERROR("Failed to read star kdtree from file %s", startreefname);
        goto bailout;
	}
	free(startreefname);
    startreefname = NULL;
	index->meta.index_jitter = qfits_header_getdouble(index->starkd->header, "JITTER", DEFAULT_INDEX_JITTER);

	if (flags & INDEX_ONLY_LOAD_SKDT)
		return index;

	// Read .quad file...
	logverb("Reading quads file %s...\n", quadfname);
    gettimeofday(&tv1, NULL);
	index->quads = quadfile_open(quadfname);
    gettimeofday(&tv2, NULL);
    debug("reading quad took %g ms.\n", millis_between(&tv1, &tv2));
	if (!index->quads) {
        ERROR("Failed to read quads file from %s", quadfname);
        goto bailout;
	}
	free(quadfname);
    quadfname = NULL;
	index->meta.index_scale_upper = quadfile_get_index_scale_upper_arcsec(index->quads);
	index->meta.index_scale_lower = quadfile_get_index_scale_lower_arcsec(index->quads);
	index->meta.indexid = index->quads->indexid;
	index->meta.healpix = index->quads->healpix;
	index->meta.hpnside = index->quads->hpnside;
	index->meta.dimquads = index->quads->dimquads;
	index->meta.nquads = index->quads->numquads;
	index->meta.nstars = index->quads->numstars;

	logverb("Index scale: [%g, %g] arcmin, [%g, %g] arcsec\n",
            index->meta.index_scale_lower / 60.0, index->meta.index_scale_upper / 60.0,
            index->meta.index_scale_lower, index->meta.index_scale_upper);
    logverb("Index has %i quads and %i stars\n", index->meta.nquads, index->meta.nstars);

	// Read .ckdt file...
	logverb("Reading code KD tree from %s...\n", codetreefname);
    gettimeofday(&tv1, NULL);
	index->codekd = codetree_open(codetreefname);
    gettimeofday(&tv2, NULL);
    debug("reading ckdt took %g ms.\n", millis_between(&tv1, &tv2));
	if (!index->codekd) {
		ERROR("Failed to read code kdtree from file %s", codetreefname);
        goto bailout;
    }
	free(codetreefname);
    codetreefname = NULL;

	// check for CIRCLE field in ckdt header...
	index->meta.circle = qfits_header_getboolean(index->codekd->header, "CIRCLE", 0);
	if (!index->meta.circle) {
		ERROR("Code kdtree does not contain the CIRCLE header.");
        goto bailout;
	}

	// New indexes are cooked such that cx < dx for all codes, but not
	// all of the old ones are like this.
    index->meta.cx_less_than_dx = qfits_header_getboolean(index->codekd->header, "CXDX", FALSE);

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

void index_close(index_t* index) {
	if (!index) return;
	free(index->meta.indexname);
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
