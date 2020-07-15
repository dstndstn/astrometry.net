/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "kdtree.h"
#include "starutil.h"
#include "quadfile.h"
#include "fitsioutils.h"
#include "codekd.h"
#include "anqfits.h"
#include "log.h"
#include "errors.h"
#include "boilerplate.h"

int unpermute_quads(quadfile_t* quadin, codetree_t* treein,
                    quadfile_t* quadout, codetree_t** p_treeout,
                    char** args, int argc) {
    int i;
    qfits_header* codehdr;
    qfits_header* hdr;
    int healpix;
    int hpnside;
    int codehp = -1;
    qfits_header* qouthdr;
    qfits_header* qinhdr;
    codetree_t* treeout;
    anbool allsky;

    codehdr = codetree_header(treein);
    healpix = quadin->healpix;
    hpnside = quadin->hpnside;

    allsky = qfits_header_getboolean(codehdr, "ALLSKY", 0);
    if (allsky)
        logverb("Index is all-sky\n");
    else {
        codehp = qfits_header_getint(codehdr, "HEALPIX", -1);
        if (codehp == -1)
            ERROR("Warning, input code kdtree didn't have a HEALPIX header");
        else if (codehp != healpix) {
            ERROR("Quadfile says it's healpix %i, but code kdtree says %i",
                  healpix, codehp);
            return -1;
        }
    }

    quadout->healpix = healpix;
    quadout->hpnside = hpnside;
    quadout->indexid = quadin->indexid;
    quadout->numstars = quadin->numstars;
    quadout->dimquads = quadin->dimquads;
    quadout->index_scale_upper = quadin->index_scale_upper;
    quadout->index_scale_lower = quadin->index_scale_lower;

    qouthdr = quadfile_get_header(quadout);
    qinhdr  = quadfile_get_header(quadin);

    BOILERPLATE_ADD_FITS_HEADERS(qouthdr);
    qfits_header_add(qouthdr, "HISTORY", "This file was created by the program \"unpermute-quads\".", NULL, NULL);
    qfits_header_add(qouthdr, "HISTORY", "unpermute-quads command line:", NULL, NULL);
    fits_add_args(qouthdr, args, argc);
    qfits_header_add(qouthdr, "HISTORY", "(end of unpermute-quads command line)", NULL, NULL);
    qfits_header_add(qouthdr, "HISTORY", "** unpermute-quads: history from input:", NULL, NULL);
    fits_copy_all_headers(qinhdr, qouthdr, "HISTORY");
    qfits_header_add(qouthdr, "HISTORY", "** unpermute-quads end of history from input.", NULL, NULL);
    qfits_header_add(qouthdr, "COMMENT", "** unpermute-quads: comments from input:", NULL, NULL);
    fits_copy_all_headers(qinhdr, qouthdr, "COMMENT");
    qfits_header_add(qouthdr, "COMMENT", "** unpermute-quads: end of comments from input.", NULL, NULL);
    an_fits_copy_header(qinhdr, qouthdr, "CXDX");
    an_fits_copy_header(qinhdr, qouthdr, "CXDXLT1");
    an_fits_copy_header(qinhdr, qouthdr, "CIRCLE");
    an_fits_copy_header(qinhdr, qouthdr, "ALLSKY");

    if (quadfile_write_header(quadout)) {
        ERROR("Failed to write quadfile header");
        return -1;
    }

    for (i=0; i<codetree_N(treein); i++) {
        unsigned int stars[quadin->dimquads];
        int ind = codetree_get_permuted(treein, i);
        if (quadfile_get_stars(quadin, ind, stars)) {
            ERROR("Failed to read quad entry");
            return -1;
        }
        if (quadfile_write_quad(quadout, stars)) {
            ERROR("Failed to write quad entry");
            return -1;
        }
    }

    if (quadfile_fix_header(quadout)) {
        ERROR("Failed to fix quadfile header");
        return -1;
    }

    treeout = codetree_new();
    treeout->tree = malloc(sizeof(kdtree_t));
    memcpy(treeout->tree, treein->tree, sizeof(kdtree_t));
    treeout->tree->perm = NULL;

    hdr = codetree_header(treeout);
    an_fits_copy_header(qinhdr, hdr, "HEALPIX");
    an_fits_copy_header(qinhdr, hdr, "HPNSIDE");
    an_fits_copy_header(qinhdr, hdr, "ALLSKY");
    BOILERPLATE_ADD_FITS_HEADERS(hdr);
    qfits_header_add(hdr, "HISTORY", "This file was created by the program \"unpermute-quads\".", NULL, NULL);
    qfits_header_add(hdr, "HISTORY", "unpermute-quads command line:", NULL, NULL);
    fits_add_args(hdr, args, argc);
    qfits_header_add(hdr, "HISTORY", "(end of unpermute-quads command line)", NULL, NULL);
    qfits_header_add(hdr, "HISTORY", "** unpermute-quads: history from input ckdt:", NULL, NULL);
    fits_copy_all_headers(codehdr, hdr, "HISTORY");
    qfits_header_add(hdr, "HISTORY", "** unpermute-quads end of history from input ckdt.", NULL, NULL);
    qfits_header_add(hdr, "COMMENT", "** unpermute-quads: comments from input ckdt:", NULL, NULL);
    fits_copy_all_headers(codehdr, hdr, "COMMENT");
    qfits_header_add(hdr, "COMMENT", "** unpermute-quads: end of comments from input ckdt.", NULL, NULL);
    an_fits_copy_header(codehdr, hdr, "CXDX");
    an_fits_copy_header(codehdr, hdr, "CXDXLT1");
    an_fits_copy_header(codehdr, hdr, "CIRCLE");

    *p_treeout = treeout;
    return 0;
}

int unpermute_quads_files(const char* quadinfn, const char* ckdtinfn,
                          const char* quadoutfn, const char* ckdtoutfn,
                          char** args, int argc) {
    quadfile_t* quadin;
    quadfile_t* quadout;
    codetree_t* treein;
    codetree_t* treeout;

    logmsg("Reading code tree from %s ...\n", ckdtinfn);
    treein = codetree_open(ckdtinfn);
    if (!treein) {
        ERROR("Failed to read code kdtree from %s", ckdtinfn);
        return -1;
    }

    logmsg("Reading quads from %s ...\n", quadinfn);
    quadin = quadfile_open(quadinfn);
    if (!quadin) {
        ERROR("Failed to read quads from %s", quadinfn);
        return -1;
    }

    logmsg("Writing quads to %s ...\n", quadoutfn);
    quadout = quadfile_open_for_writing(quadoutfn);
    if (!quadout) {
        ERROR("Failed to write quads to %s", quadoutfn);
        return -1;
    }

    if (unpermute_quads(quadin, treein, quadout, &treeout, args, argc)) {
        return -1;
    }

    if (quadfile_close(quadout)) {
        ERROR("Failed to close output quadfile");
        return -1;
    }

    quadfile_close(quadin);

    logmsg("Writing code kdtree to %s ...\n", ckdtoutfn);
    if (codetree_write_to_file(treeout, ckdtoutfn) ||
        codetree_close(treeout)) {
        ERROR("Failed to write code kdtree");
        return -1;
    }

    free(treein->tree);
    treein->tree = NULL;
    codetree_close(treein);
    return 0;
}
