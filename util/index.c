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

char* index_get_qidx_filename(const char* indexname) {
    char* quadfn;
    char* qidxfn = NULL;
    bool singlefile;
    if (!index_is_file_index(indexname))
        return NULL;
    get_filenames(indexname, &quadfn, NULL, NULL, &singlefile);
    if (singlefile) {
        if (ends_with(quadfn, ".fits")) {
            asprintf(&qidxfn, "%.*s.qidx.fits", (int)(strlen(quadfn)-5), quadfn);
        } else {
            asprintf(&qidxfn, "%s.qidx.fits", quadfn);
        }
    } else {
        if (ends_with(quadfn, ".quad.fits")) {
            asprintf(&qidxfn, "%.*s.qidx.fits", (int)(strlen(quadfn)-10), quadfn);
        } else {
            asprintf(&qidxfn, "%s.qidx.fits", quadfn);
        }
    }
    free(quadfn);
    return qidxfn;
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

int index_get_missing_cut_params(int indexid, int* hpnside, int* nsweep,
								 double* dedup, int* margin, char** pband) {
	// The 200-series indices use cut 100 (usnob)
	// The 500-series indices use cut 100 (usnob)
	// The 600-series indices use cut 300 (2mass)
	// The 700-series indices use cut 400 (usnob)
	int i = -1;
	int ns, n, marg;
	double dd;
	char* band;

	if ((indexid >= 200 && indexid < 220) ||
		(indexid >= 500 && indexid < 520)) {
		// Cut 100 params:
		int cut100hp[] = { 1760, 1245, 880, 622, 440, 312, 220, 156, 110, 78, 55, 39, 28, 20, 14, 10, 7, 5, 4, 3 };
		int cut100n[] = { 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
		double cut100dd[] = { 8, 8, 8, 8, 8, 9.6, 13.2, 18.0, 25.2, 36, 51, 72, 102, 144, 204, 288, 408, 600, 840, 1200 };
		int cut100margin = 5;

		i = indexid % 100;
		ns = cut100hp[i];
		n = cut100n[i];
		dd = cut100dd[i];
		marg = cut100margin;
		band = "R";

	} else if (indexid >= 602 && indexid < 620) {
		// Cut 300 params:
		int cut300hp[] = { 0, 0, 880, 624, 440, 312, 220, 156, 110, 78, 56, 40, 28, 20, 14, 10, 8, 6, 4, 4 };
		int cut300n = 10;
		double cut300dd = 8.0;
		//double cut300dd[] = { 8, 8, 8, 8, 8, 9.6, 13.2, 18.0, 25.2, 36, 51, 72, 102, 144, 204, 288, 408, 600, 840, 1200 };
		int cut300margin = 10;

		i = indexid % 100;
		ns = cut300hp[i];
		n = cut300n;
		dd = cut300dd;
		marg = cut300margin;
		band = "J";

	} else if (indexid >= 700 && indexid < 720) {
		// Cut 400 params:
		// (cut 400 used cut 200 as input: it had dedup=8, and n=6,10,10,...)
		int cut400hp[] = { 1760, 1246, 880, 624, 440, 312, 220, 156, 110, 78, 55, 39, 28, 20, 14, 10, 7, 5, 4, 3 };
		int cut400n[] = { 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };
		double cut400dd = 8.0;
		//double cut400dd[] = { 8, 8, 8, 8, 8, 9.6, 13.2, 18.0, 25.2, 36, 51, 72, 102, 144, 204, 288, 408, 600, 840, 1200 };
		int cut400margin = 10;

		i = indexid % 100;
		ns = cut400hp[i];
		n = cut400n[i];
		dd = cut400dd;
		marg = cut400margin;
		band = "R";
	} else {
		return -1;
	}

	if (hpnside)
		*hpnside = ns;
	if (nsweep)
		*nsweep = n;
	if (dedup)
		*dedup = dd;
	if (margin)
		*margin = marg;
	if (pband)
		*pband = strdup(band);
	return 0;
}

static void get_cut_params(index_t* index) {
	index_meta_t* meta = &(index->meta);
	meta->index_jitter = startree_get_jitter(index->starkd);
	if (meta->index_jitter == 0.0)
		meta->index_jitter = DEFAULT_INDEX_JITTER;

	meta->cutnside = startree_get_cut_nside(index->starkd);
	meta->cutnsweep = startree_get_cut_nsweeps(index->starkd);
	meta->cutdedup = startree_get_cut_dedup(index->starkd);
	meta->cutband = strdup_safe(startree_get_cut_band(index->starkd));
	meta->cutmargin = startree_get_cut_margin(index->starkd);

	// HACK - fill in values that are missing in old index files.
	{
		int *nside = NULL, *nsweep = NULL, *margin = NULL;
		char** band = NULL;
		double* dedup = NULL;

		if (meta->cutnside == -1)
			nside = &(meta->cutnside);
		if (meta->cutnsweep == 0)
			nsweep = &(meta->cutnsweep);
		if (meta->cutmargin == -1)
			margin = &(meta->cutmargin);
		if (meta->cutdedup == 0)
			dedup = &(meta->cutdedup);
		if (!meta->cutband)
			band = &(meta->cutband);

		index_get_missing_cut_params(meta->indexid, nside, nsweep, dedup, margin, band);
	}

}

index_t* index_load(const char* indexname, int flags) {
    struct timeval tv1, tv2;

	char *codetreefname=NULL, *quadfname=NULL, *startreefname=NULL;
    bool singlefile;
	index_t* index = calloc(1, sizeof(index_t));

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

	// This must get called after meta.indexid is set: otherwise we won't be
	// able to fill in values that are missing in old index files.
	get_cut_params(index);

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
	free(index->meta.cutband);
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
