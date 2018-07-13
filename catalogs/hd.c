/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>

#include "kdtree_fits_io.h"
#include "kdtree.h"
#include "errors.h"
#include "starutil.h"
#include "hd.h"

hd_catalog_t* henry_draper_open(const char* fn) {
    hd_catalog_t* hd = calloc(1, sizeof(hd_catalog_t));
    hd->fn = strdup(fn);
    hd->kd = kdtree_fits_read(hd->fn, NULL, NULL);
    if (!hd->kd) {
        ERROR("Failed to read a kdtree from file %s", hd->fn);
        return NULL;
    }
    return hd;
}

int henry_draper_n(const hd_catalog_t* hd) {
    assert(hd);
    assert(hd->kd);
    return kdtree_n(hd->kd);
}

void henry_draper_close(hd_catalog_t* hd) {
    if (!hd) return;
    free(hd->fn);
    kdtree_fits_close(hd->kd);
    free(hd);
}

bl* henry_draper_get(hd_catalog_t* hdcat,
                     double racenter, double deccenter,
                     double r_arcsec) {
    double r2;
    double xyz[3];
    kdtree_qres_t* q;
    bl* res;
    int i;
    hd_entry_t hd;

    radecdeg2xyzarr(racenter, deccenter, xyz);
    r2 = arcsec2distsq(r_arcsec);
    q = kdtree_rangesearch(hdcat->kd, xyz, r2);
    if (!q) {
        return NULL;
    }

    res = bl_new(256, sizeof(hd_entry_t));
    for (i=0; i<q->nres; i++) {
        double* pt = q->results.d + i*3;
        xyzarr2radecdeg(pt, &(hd.ra), &(hd.dec));
        hd.hd = q->inds[i] + 1;
        bl_append(res, &hd);
    }

    kdtree_free_query(q);

    return res;
}

