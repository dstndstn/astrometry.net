/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "cutest.h"
#include "multiindex.h"
#include "bl.h"

/**
 fitscopy ~/DATA/tycho2-cut.fits"[RA<10 && DEC > 0 && DEC < 10]" t.fits
 build-index -i t.fits -o t10.index -P 10 -E -M -v -S mag
 build-index -1 t10.index -o t11.index -P 11 -E -M -v -S mag
 build-index -1 t10.index -o t12.index -P 12 -E -M -v -S mag
 fitsgetext -i t10.index -o t10.skdt -e 0 -e 7 -e 8 -e 9 -e 10 -e 11 -e 12 -e 13
 fitsgetext -i t10.index -o t10.ind -e 0 -e 1 -e 2 -e 3 -e 4 -e 5 -e 6
 fitsgetext -i t11.index -o t11.ind -e 0 -e 1 -e 2 -e 3 -e 4 -e 5 -e 6
 fitsgetext -i t12.index -o t12.ind -e 0 -e 1 -e 2 -e 3 -e 4 -e 5 -e 6
 */
void test_multiindex(CuTest* ct) {
    sl* fns;
    multiindex_t* mi;
    int i;

    fns = sl_new(4);
    sl_append(fns, "t10.ind");
    sl_append(fns, "t11.ind");
    sl_append(fns, "t12.ind");
    mi = multiindex_open("t10.skdt", fns, 0);

    printf("Got %i indices\n", multiindex_n(mi));
    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        printf("  %i: %s, %i stars, %i quads (%g to %g arcmin)\n",
               i, ind->indexname, index_nquads(ind), index_nstars(ind),
               ind->index_scale_lower/60., ind->index_scale_upper/60.);
    }

    multiindex_free(mi);
    sl_free2(fns);
}


