/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

#include "cutest.h"
#include "hd.h"
#include "starutil.h"
#include "ioutils.h"

void test_hd_1(CuTest* tc) {
    hd_catalog_t* hdcat;
    int* invperm;
    int* present;
    int ind, i, N;
    double xyz[3];
    double ra, dec;
    int strangehds[] = { 40142, 40441, 40672, 40746, 40763, 40764,
                         104176, 104193, 163635, 224698, 224699,
                         129371 };

    if (!file_readable("hd.fits")) {
        printf("File \"hd.fits\" does not exist; test skipped.\n");
        return;
    }

    hdcat = henry_draper_open("hd.fits");
    CuAssertPtrNotNull(tc, hdcat);

    N = hdcat->kd->ndata;
    invperm = calloc(N, sizeof(int));
    CuAssertPtrNotNull(tc, invperm);

    CuAssertIntEquals(tc, 0, kdtree_check(hdcat->kd));

    kdtree_inverse_permutation(hdcat->kd, invperm);

    present = calloc(N, sizeof(int));
    for (i=0; i<N; i++) {
        CuAssert(tc, "invperm in range", invperm[i] < N);
        present[invperm[i]]++;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(tc, 1, present[i]);
    }
    free(present);

    // Where is "HD n" ?
    for (i=0; i<10; i++) {
        bl* res;
        int j;

        ind = invperm[i];
        kdtree_copy_data_double(hdcat->kd, ind, 1, xyz);
        xyzarr2radecdeg(xyz, &ra, &dec);
        printf("HD %i: RA,Dec %g, %g\n", i+1, ra, dec);

        res = henry_draper_get(hdcat, ra, dec, 10.0);
        CuAssertPtrNotNull(tc, res);
        for (j=0; j<bl_size(res); j++) {
            hd_entry_t* hd = bl_access(res, j);
            printf("res %i: HD %i, RA, Dec %g, %g\n", j, hd->hd, hd->ra, hd->dec);
        }
        bl_free(res);
    }

    for (i=0; i<sizeof(strangehds)/sizeof(int); i++) {
        ind = invperm[strangehds[i]-1];
        kdtree_copy_data_double(hdcat->kd, ind, 1, xyz);
        xyzarr2radecdeg(xyz, &ra, &dec);
        printf("HD %i: RA,Dec %g, %g\n", strangehds[i], ra, dec);
    }
    free(invperm);

    henry_draper_close(hdcat);
}
