/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "kdtree.h"
#include "starutil.h"
#include "quadfile.h"
#include "fitsioutils.h"
#include "anqfits.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"

int unpermute_stars(startree_t* treein, quadfile_t* qfin,
                    startree_t** p_treeout, quadfile_t* qfout,
                    anbool dosweeps, anbool check,
                    char** args, int argc) {
    startree_t* treeout;
    int i;
    int N;
    int healpix = -1;
    int hpnside = 0;
    int starhp = -1;
    int lastgrass;
    qfits_header* qouthdr;
    qfits_header* qinhdr;
    anbool allsky;

    assert(p_treeout);
    N = startree_N(treein);
    allsky = qfits_header_getboolean(startree_header(treein), "ALLSKY", 0);
    if (allsky)
        logverb("Star kd-tree is all-sky\n");
    else {
        starhp = qfits_header_getint(startree_header(treein), "HEALPIX", -1);
        if (starhp == -1)
            ERROR("Warning, input star kdtree didn't have a HEALPIX header.\n");
        hpnside = qfits_header_getint(startree_header(treein), "HPNSIDE", 1);
        healpix = starhp;
        logverb("Star kd-tree covers healpix %i, nside %i\n", healpix, hpnside);
    }

    qfout->healpix = healpix;
    qfout->hpnside = hpnside;
    qfout->numstars          = qfin->numstars;
    qfout->dimquads          = qfin->dimquads;
    qfout->index_scale_upper = qfin->index_scale_upper;
    qfout->index_scale_lower = qfin->index_scale_lower;
    qfout->indexid           = qfin->indexid;

    qouthdr = quadfile_get_header(qfout);
    qinhdr  = quadfile_get_header(qfin);

    an_fits_copy_header(qinhdr, qouthdr, "ALLSKY");

    BOILERPLATE_ADD_FITS_HEADERS(qouthdr);
    qfits_header_add(qouthdr, "HISTORY", "This file was created by the program \"unpermute-stars\".", NULL, NULL);
    qfits_header_add(qouthdr, "HISTORY", "unpermute-stars command line:", NULL, NULL);
    fits_add_args(qouthdr, args, argc);
    qfits_header_add(qouthdr, "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
    qfits_header_add(qouthdr, "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
    fits_copy_all_headers(qinhdr, qouthdr, "HISTORY");
    qfits_header_add(qouthdr, "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
    qfits_header_add(qouthdr, "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
    fits_copy_all_headers(qinhdr, qouthdr, "COMMENT");
    qfits_header_add(qouthdr, "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);

    if (quadfile_write_header(qfout)) {
        ERROR("Failed to write quadfile header.\n");
        return -1;
    }

    logmsg("Writing quads...\n");

    startree_compute_inverse_perm(treein);

    if (check) {
        logmsg("Running quadfile_check()...\n");
        if (quadfile_check(qfin)) {
            ERROR("quadfile_check() failed");
            return -1;
        }
        logmsg("Check passed.\n");

        logmsg("Checking inverse permutation...\n");
        if (startree_check_inverse_perm(treein)) {
            ERROR("check failed!");
            return -1;
        }

        logmsg("Running startree kdtree_check()...\n");
        if (kdtree_check(treein->tree)) {
            ERROR("kdtree_check() failed");
            return -1;
        }
        logmsg("Check passed.\n");
    }


    lastgrass = 0;
    for (i=0; i<qfin->numquads; i++) {
        int j;
        unsigned int stars[qfin->dimquads];
        if (i*80/qfin->numquads != lastgrass) {
            logmsg(".");
            fflush(stdout);
            lastgrass = i*80/qfin->numquads;
        }
        if (quadfile_get_stars(qfin, i, stars)) {
            ERROR("Failed to read quadfile entry.\n");
            return -1;
        }
        for (j=0; j<qfin->dimquads; j++)
            stars[j] = treein->inverse_perm[stars[j]];
        if (quadfile_write_quad(qfout, stars)) {
            ERROR("Failed to write quadfile entry.\n");
            return -1;
        }
    }
    logmsg("\n");


    if (quadfile_fix_header(qfout)) {
        ERROR("Failed to fix quadfile header");
        return -1;
    }

    treeout = startree_new();
    treeout->tree = malloc(sizeof(kdtree_t));
    memcpy(treeout->tree, treein->tree, sizeof(kdtree_t));
    treeout->tree->perm = NULL;

    an_fits_copy_header(startree_header(treein), startree_header(treeout), "HEALPIX");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "HPNSIDE");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "ALLSKY");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "JITTER");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTNSIDE");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTMARG");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTBAND");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTDEDUP");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTNSWEP");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTMINMG");
    an_fits_copy_header(startree_header(treein), startree_header(treeout), "CUTMAXMG");

    qfits_header_add(startree_header(treeout), "HISTORY", "unpermute-stars command line:", NULL, NULL);
    fits_add_args(startree_header(treeout), args, argc);
    qfits_header_add(startree_header(treeout), "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
    qfits_header_add(startree_header(treeout), "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
    fits_copy_all_headers(startree_header(treein), startree_header(treeout), "HISTORY");
    qfits_header_add(startree_header(treeout), "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
    qfits_header_add(startree_header(treeout), "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
    fits_copy_all_headers(startree_header(treein), startree_header(treeout), "COMMENT");
    qfits_header_add(startree_header(treeout), "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);

    if (dosweeps) {
        // copy sweepX headers.
        for (i=1;; i++) {
            char key[16];
            int n;
            sprintf(key, "SWEEP%i", i);
            n = qfits_header_getint(treein->header, key, -1);
            if (n == -1)
                break;
            an_fits_copy_header(treein->header, treeout->header, key);
        }

        // compute sweep array.
        treeout->sweep = malloc(N * sizeof(uint8_t));
        for (i=0; i<N; i++) {
            int ind = treein->tree->perm[i];
            // Stars are sorted first by sweep and then by brightness within
            // the sweep.  Instead of just storing the sweep number, we can
            // store a quantization of the total-ordered rank.
            treeout->sweep[i] = (uint8_t)floor((float)256.0 * (float)ind / (float)N);
        }
    }

    *p_treeout = treeout;
    return 0;
}

int unpermute_stars_tagalong(startree_t* treein,
                             fitstable_t* tagout) {
    fitstable_t* tagin;
    qfits_header* tmphdr;
    int N;
    tagin = startree_get_tagalong(treein);
    if (!tagin) {
        ERROR("No input tag-along table");
        return -1;
    }
    N = startree_N(treein);
    assert(fitstable_nrows(tagin) == N);
    fitstable_clear_table(tagin);
    fitstable_add_fits_columns_as_struct(tagin);
    fitstable_copy_columns(tagin, tagout);
    tmphdr = tagout->header;
    tagout->header = tagin->header;
    if (fitstable_write_header(tagout)) {
        ERROR("Failed to write tag-along table header");
        return -1;
    }
    if (fitstable_copy_rows_data(tagin, (int*)treein->tree->perm, N, tagout)) {
        ERROR("Failed to copy tag-along table rows from input to output");
        return -1;
    }
    if (fitstable_fix_header(tagout)) {
        ERROR("Failed to fix tag-along table header");
        return -1;
    }
    tagout->header = tmphdr;
    return 0;
}

int unpermute_stars_files(const char* skdtinfn, const char* quadinfn,
                          const char* skdtoutfn, const char* quadoutfn,
                          anbool dosweeps, anbool check,
                          char** args, int argc) {
    quadfile_t* qfin;
    quadfile_t* qfout;
    startree_t* treein;
    startree_t* treeout;
    fitstable_t* tagout = NULL;
    fitstable_t* tagin;
    int rtn;

    logmsg("Reading star tree from %s ...\n", skdtinfn);
    treein = startree_open(skdtinfn);
    if (!treein) {
        ERROR("Failed to read star kdtree from %s.\n", skdtinfn);
        return -1;
    }

    logmsg("Reading quadfile from %s ...\n", quadinfn);
    qfin = quadfile_open(quadinfn);
    if (!qfin) {
        ERROR("Failed to read quadfile from %s.\n", quadinfn);
        return -1;
    }

    logmsg("Writing quadfile to %s ...\n", quadoutfn);
    qfout = quadfile_open_for_writing(quadoutfn);
    if (!qfout) {
        ERROR("Failed to write quadfile to %s.\n", quadoutfn);
        return -1;
    }

    rtn = unpermute_stars(treein, qfin, &treeout, qfout,
                          dosweeps, check, args, argc);
    if (rtn)
        return rtn;

    if (quadfile_close(qfout)) {
        ERROR("Failed to close output quadfile.\n");
        return -1;
    }

    logmsg("Writing star kdtree to %s ...\n", skdtoutfn);
    if (startree_write_to_file(treeout, skdtoutfn)) {
        ERROR("Failed to write star kdtree.\n");
        return -1;
    }

    if (startree_has_tagalong(treein)) {
        logmsg("Permuting tag-along table...\n");
        tagin = startree_get_tagalong(treein);
        if (tagin) {
            tagout = fitstable_open_for_appending(skdtoutfn);
            tagout->table = fits_copy_table(tagin->table);
            tagout->table->nr = 0;
            if (unpermute_stars_tagalong(treein, tagout)) {
                ERROR("Failed to permute tag-along table");
                return -1;
            }
            if (fitstable_close(tagout)) {
                ERROR("Failed to close tag-along data");
                return -1;
            }
        }
    }

    quadfile_close(qfin);
    startree_close(treein);
    free(treeout->sweep);
    free(treeout->tree);
    treeout->tree = NULL;
    startree_close(treeout);

    return 0;
}
