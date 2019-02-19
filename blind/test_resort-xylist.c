/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cutest.h"
#include "xylist.h"
#include "starxy.h"
#include "resort-xylist.h"

void test_sorting(CuTest* tc) {
    double flux[] = { 50, 100, 50, 100, 20, 20, 40, 40 };
    double bg[]   = {  0,  10, 10,   0, 10,  0,  5,  0 };

    int trueorder[] = { 4, 5, 7, 6, 2, 0, 1, 3 };

    int i, N;
    starxy_t* s;
    char* infn = "/tmp/test-resort-xylist";
    char* outfn = "/tmp/test-resort-xylist-out";
    xylist_t* xy;

    xylist_t* xy2;
    starxy_t* s2;

    xy = xylist_open_for_writing(infn);
    CuAssertPtrNotNull(tc, xy);
    xylist_set_include_flux(xy, TRUE);
    xylist_set_include_background(xy, TRUE);
    if (xylist_write_primary_header(xy) ||
        xylist_write_header(xy)) {
        CuFail(tc, "write header");
    }

    N = sizeof(flux) / sizeof(double);
    s = starxy_new(N, TRUE, TRUE);

    for (i=0; i<N; i++) {
        starxy_setx(s, i, random()%1000);
        starxy_sety(s, i, random()%1000);
    }
    starxy_set_flux_array(s, flux);
    starxy_set_bg_array(s, bg);
    if (xylist_write_field(xy, s) ||
        xylist_fix_header(xy) ||
        xylist_fix_primary_header(xy) ||
        xylist_close(xy)) {
        CuFail(tc, "close xy");
    }

    CuAssertIntEquals(tc, 0,
                      resort_xylist(infn, outfn, NULL, NULL, TRUE));

    xy2 = xylist_open(outfn);
    s2 = xylist_read_field(xy2, NULL);
    CuAssertPtrNotNull(tc, s2);

    CuAssertPtrNotNull(tc, s2->x);
    CuAssertPtrNotNull(tc, s2->y);
    CuAssertPtrNotNull(tc, s2->flux);
    CuAssertPtrNotNull(tc, s2->background);

    for (i=0; i<N; i++) {
        fflush(NULL);
        printf("Testing point %i\n", i);
        fflush(NULL);

        CuAssertDblEquals(tc, s->x[trueorder[i]], s2->x[i], 1e-6);
        CuAssertDblEquals(tc, s->y[trueorder[i]], s2->y[i], 1e-6);
        CuAssertDblEquals(tc, s->flux[trueorder[i]], s2->flux[i], 1e-6);
        CuAssertDblEquals(tc, s->background[trueorder[i]], s2->background[i], 1e-6);
    }
}
