/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>

#include "cutest.h"
#include "multiindex.h"
#include "solver.h"
#include "xylist.h"
#include "bl.h"
#include "log.h"
#include "errors.h"


/**

 fitscopy ~/DATA/tycho2-cut.fits"[RA<10 && DEC > 0 && DEC < 10]" t.fits
 build-index -i t.fits -o t10.index -P 10 -E -M -v -S mag
 build-index -1 t10.index -o t11.index -P 11 -E -M -v -S mag
 build-index -1 t10.index -o t12.index -P 12 -E -M -v -S mag
 fitsgetext -i t10.index -o t10.skdt -e 0 -e 7 -e 8 -e 9 -e 10 -e 11 -e 12 -e 13
 fitsgetext -i t10.index -o t10.ind -e 0 -e 1 -e 2 -e 3 -e 4 -e 5 -e 6
 fitsgetext -i t11.index -o t11.ind -e 0 -e 1 -e 2 -e 3 -e 4 -e 5 -e 6
 fitsgetext -i t12.index -o t12.ind -e 0 -e 1 -e 2 -e 3 -e 4 -e 5 -e 6

 make-wcs.py -r 5 -d 5 -s 3 -W 1000 -H 1000 t1.wcs
 query-starkd -o t10.rd -r 5 -d 5 -R 1.5 -t mag util/t10.skdt 
 wcs-rd2xy -w t1.wcs -i t10.rd -o t1.xy
 solve-field --just-augment --scale-low 1 t1.xy --continue --width 1000 --height 1000
 backend -c none -i 't1?.index' -v t1.axy 

 (in util/ directory:)
 ../solver/test_multiindex2

 */

void test_solve_multiindex(CuTest* ct) {
    sl* fns;
    multiindex_t* mi;
    int i;
    solver_t* s = NULL;
    starxy_t* field = NULL;
    MatchObj* mo = NULL;
    xylist_t* xy = NULL;

    log_init(LOG_VERB);

    fns = sl_new(4);
    sl_append(fns, "../util/t10.ind");
    sl_append(fns, "../util/t11.ind");
    sl_append(fns, "../util/t12.ind");
    mi = multiindex_open("../util/t10.skdt", fns, 0);

    printf("Got %i indices\n", multiindex_n(mi));
    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        printf("  %i: %s, %i stars, %i quads (%g to %g arcmin)\n",
               i, ind->indexname, index_nquads(ind), index_nstars(ind),
               ind->index_scale_lower/60., ind->index_scale_upper/60.);
    }

    s = solver_new();

    // 10.8
    s->funits_lower = 5.0;
    s->funits_upper = 15.0;

    xy = xylist_open("../util/t1.xy");
    if (!xy) {
        ERROR("Failed to open xylist\n");
        CuFail(ct, "xylist");
    }
    field = xylist_read_field(xy, NULL);

    solver_set_field(s, field);
    solver_set_field_bounds(s, 0, 1000, 0, 1000);

    for (i=0; i<multiindex_n(mi); i++) {
        index_t* ind = multiindex_get(mi, i);
        solver_add_index(s, ind);
    }

    solver_run(s);

    if (solver_did_solve(s)) {
        mo = solver_get_best_match(s);
        matchobj_print(mo, LOG_MSG);

        // HACK -- ugly!!
        verify_free_matchobj(mo);
    }

    xylist_close(xy);

    solver_cleanup_field(s);
    solver_free(s);

    multiindex_free(mi);
    sl_free2(fns);
}

