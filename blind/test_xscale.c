/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <assert.h>

#include "cutest.h"
#include "index.h"
#include "solver.h"
#include "xylist.h"
#include "sip.h"
#include "sip-utils.h"
#include "bl.h"
#include "log.h"
#include "errors.h"
#include "sip_qfits.h"

/*
 - example image: DESI Commissioning Instrument exposure 3367-CIW.
 */

void test_xscale(CuTest* ct) {
    // core Astrometry solver parameters
    solver_t* solver;
    int imagew, imageh;
    double imagecx, imagecy;
    double arcmin_width_min = 5;
    double arcmin_width_max = 8;

    char* xyfn = "3367-W.xy";
    //char* xyfn = "3367-W-110.axy";
    char* indexfn = "/data1/INDEXES/5000/index-5001-31.fits";

    int loglvl = LOG_MSG;
    loglvl++;
    log_init(loglvl);
    
    imagew = 3072;
    imageh = 2048;

    imagecx = (imagew - 1.0)/2.0;
    imagecy = (imageh - 1.0)/2.0;
    
    // Here we initialize the core astrometry solver struct, telling
    // it about the possible range of image scales.
    solver = solver_new();
    double qsf_min = 0.1;
    // don't try teeny-tiny quads.
    solver->quadsize_min = qsf_min * MIN(imagew, imageh);

    // compute scale range in arcseconds per pixel.
    // set the solver's "funits" = field (image) scale units
    solver->funits_lower = 60. * arcmin_width_min / (double)imagew;
    solver->funits_upper = 60. * arcmin_width_max / (double)imagew;

    solver_set_keep_logodds(solver, log(1e12));
    solver->logratio_toprint = log(1e6);
    solver->endobj = 20;
    
    xylist_t* xyls = xylist_open(xyfn);
    starxy_t* xy = xylist_read_field(xyls, NULL);
    // Feed the image source coordinates to the solver...
    solver_set_field(solver, xy);
    solver_set_field_bounds(solver, 0, imagew, 0, imageh);

    index_t* index = index_load(indexfn, 0, NULL);
    solver_add_index(solver, index);
    solver->distance_from_quad_bonus = TRUE;
    solver->do_tweak = TRUE;
    solver->tweak_aborder = 1;
    solver->tweak_abporder = 4;
    solver_run(solver);

    CuAssert(ct, "Should not solve on undistorted field", !solver->best_match_solves);

    // "solver" will free the original "xy", so make a copy.
    xy = starxy_copy(xy);
    solver_set_field(solver, xy);

    solver_reset_best_match(solver);
    solver_reset_counters(solver);

    solver->pixel_xscale = 1.1;

    // *1.1?
    solver_set_field_bounds(solver, 0, imagew, 0, imageh);

    solver_print_to(solver, stdout);
    
    solver_run(solver);

    CuAssert(ct, "Should succeeded on scaled field", solver->best_match_solves);

    double ra, dec;
    double pscale;
    tan_t* wcs;
    logmsg("Solved using index %s with odds ratio %g\n",
           solver->best_index->indexname,
           solver->best_match.logodds);
    // WCS is solver->best_match.wcstan
    wcs = &(solver->best_match.wcstan);
    // center
    tan_pixelxy2radec(wcs, imagecx, imagecy, &ra, &dec);
    pscale = tan_pixel_scale(wcs);
    logmsg("Image center is RA,Dec = (%g,%g) degrees, size is %.2g x %.2g arcmin.\n",
           ra, dec, arcsec2arcmin(pscale * imagew), arcsec2arcmin(pscale * imageh));

    printf("Writing SIP solution...\n");
    //solver->best_match.wcstan;
    sip_write_to_file(solver->best_match.sip, "scaled-sip.wcs");

}

