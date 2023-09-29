/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdarg.h>

#include "os-features.h"
#include "ioutils.h"
#include "mathutil.h"
#include "matchobj.h"
#include "solver.h"
#include "verify.h"
#include "tic.h"
#include "solvedfile.h"
#include "fit-wcs.h"
#include "sip-utils.h"
#include "keywords.h"
#include "log.h"
#include "pquad.h"
#include "kdtree.h"
#include "quad-utils.h"
#include "errors.h"
#include "tweak2.h"

#if TESTING_TRYALLCODES
#define DEBUGSOLVER 1
#define TRY_ALL_CODES test_try_all_codes
void test_try_all_codes(pquad* pq,
                        int* fieldstars, int dimquad,
                        solver_t* solver, double tol2);

#else
#define TRY_ALL_CODES try_all_codes
#endif

#if TESTING_TRYPERMUTATIONS
#define DEBUGSOLVER 1
#define TEST_TRY_PERMUTATIONS test_try_permutations
void test_try_permutations(int* stars, double* code, int dimquad, solver_t* s);

#else
#define TEST_TRY_PERMUTATIONS(u,v,x,y)  // no-op.
#endif






void solver_set_keep_logodds(solver_t* solver, double logodds) {
    solver->logratio_tokeep = logodds;
}

int solver_set_parity(solver_t* solver, int parity) {
    if (!((parity == PARITY_NORMAL) || (parity == PARITY_FLIP) || (parity == PARITY_BOTH))) {
        ERROR("Invalid parity value: %i", parity);
        return -1;
    }
    solver->parity = parity;
    return 0;
}

anbool solver_did_solve(const solver_t* solver) {
    return solver->best_match_solves;
}

void solver_get_quad_size_range_arcsec(const solver_t* solver, double* qmin, double* qmax) {
    if (qmin) {
        *qmin = solver->quadsize_min * solver_get_pixscale_low(solver);
    }
    if (qmax) {
        double q = solver->quadsize_max;
        if (q == 0)
            q = solver->field_diag;
        *qmax = q * solver_get_pixscale_high(solver);
    }
}

double solver_get_field_jitter(const solver_t* solver) {
    return solver->verify_pix;
}

void solver_get_field_center(const solver_t* solver, double* px, double* py) {
    if (px)
        *px = (solver->field_maxx + solver->field_minx)/2.0;
    if (py)
        *py = (solver->field_maxy + solver->field_miny)/2.0;
}

double solver_get_max_radius_arcsec(const solver_t* solver) {
    return solver->funits_upper * solver->field_diag / 2.0;
}

MatchObj* solver_get_best_match(solver_t* solver) {
    return &(solver->best_match);
}

const char* solver_get_best_match_index_name(const solver_t* solver) {
    return solver->best_index->indexname;
}

double solver_get_pixscale_low(const solver_t* solver) {
    return solver->funits_lower;
}
double solver_get_pixscale_high(const solver_t* solver) {
    return solver->funits_upper;
}

void solver_set_quad_size_range(solver_t* solver, double qmin, double qmax) {
    solver->quadsize_min = qmin;
    solver->quadsize_max = qmax;
}

void solver_set_quad_size_fraction(solver_t* solver, double qmin, double qmax) {
    solver_set_quad_size_range(solver, qmin * MIN(solver_field_width(solver), solver_field_height(solver)),
                               qmax * solver->field_diag);
}

void solver_tweak2(solver_t* sp, MatchObj* mo, int order, sip_t* verifysip) {
    double* xy = NULL;
    int Nxy;
    double indexjitter;
    // quad center
    double qc[2];
    // quad radius-squared
    double Q2;
    // initial WCS
    sip_t startsip;
    int* theta;
    double* odds;
    double* refradec;
    int i;
    double newodds;
    int nm, nc, nd;
    int besti;
    int startorder;

    indexjitter = mo->index_jitter; // ref cat positional error, in arcsec.
    xy = starxy_to_xy_array(sp->fieldxy, NULL);
    Nxy = starxy_n(sp->fieldxy);
    qc[0] = (mo->quadpix[0] + mo->quadpix[2]) / 2.0;
    qc[1] = (mo->quadpix[1] + mo->quadpix[3]) / 2.0;
    Q2 = 0.25 * distsq(mo->quadpix, mo->quadpix + 2, 2);
    if (Q2 == 0.0) {
        // can happen if we're verifying an existing WCS
        // note, this is radius-squared, so 1e6 is not crazy.
        Q2 = 1e6;
        // set qc to the image center here?  or crpix?
        logverb("solver_tweak2(): setting Q2=%g; qc=(%g,%g)\n", Q2, qc[0], qc[1]);
    }

    // mo->refradec may be NULL at this point, so get it from refxyz instead...
    refradec = malloc(3 * mo->nindex * sizeof(double));
    for (i=0; i<mo->nindex; i++)
        xyzarr2radecdegarr(mo->refxyz + i*3, refradec + i*2);

    // Verifying an existing WCS?
    if (verifysip) {
        memcpy(&startsip, verifysip, sizeof(sip_t));
        startorder = MIN(verifysip->a_order, sp->tweak_aborder);
    } else {
        startorder = 1;
        sip_wrap_tan(&(mo->wcstan), &startsip);
    }

    startsip.ap_order = startsip.bp_order = sp->tweak_abporder;
    startsip.a_order = startsip.b_order = sp->tweak_aborder;
    logverb("solver_tweak2: setting orders %i, %i\n", sp->tweak_aborder, sp->tweak_abporder);

    // for TWEAK_DEBUG_PLOTs
    theta = mo->theta;
    besti = mo->nbest-1;//mo->nmatch + mo->nconflict + mo->ndistractor;

    logverb("solver_tweak2: set_crpix %i, crpix (%.1f,%.1f)\n",
            sp->set_crpix, sp->crpix[0], sp->crpix[1]);
    mo->sip = tweak2(xy, Nxy,
                     sp->verify_pix, // pixel positional noise sigma
                     solver_field_width(sp),
                     solver_field_height(sp),
                     refradec, mo->nindex,
                     indexjitter, qc, Q2,
                     sp->distractor_ratio,
                     sp->logratio_bail_threshold,
                     order, sp->tweak_abporder,
                     &startsip, NULL, &theta, &odds,
                     sp->set_crpix ? sp->crpix : NULL,
                     &newodds, &besti, mo->testperm, startorder);
    free(refradec);

    // FIXME -- update refxy?  Nobody uses it, right?
    free(mo->refxy);
    mo->refxy = NULL;
    // FIXME -- and testperm?
    free(mo->testperm);
    mo->testperm = NULL;

    if (mo->sip) {
        // Yoink the TAN solution (?)
        memcpy(&(mo->wcstan), &(mo->sip->wcstan), sizeof(tan_t));

        // Plug in the new "theta" and "odds".
        free(mo->theta);
        free(mo->matchodds);
        mo->theta = theta;
        mo->matchodds = odds;

        mo->logodds = newodds;

        verify_count_hits(theta, besti, &nm, &nc, &nd);
        mo->nmatch = nm;
        mo->nconflict = nc;
        mo->ndistractor = nd;
        matchobj_compute_derived(mo);
    }
    free(xy);
}

void solver_log_params(const solver_t* sp) {
    int i;
    logverb("Solver:\n");
    logverb("  Arcsec per pix range: %g, %g\n", sp->funits_lower, sp->funits_upper);
    logverb("  Image size: %g x %g\n", solver_field_width(sp), solver_field_height(sp));
    logverb("  Quad size range: %g, %g\n", sp->quadsize_min, sp->quadsize_max);
    logverb("  Objs: %i, %i\n", sp->startobj, sp->endobj);
    logverb("  Parity: %i, %s\n", sp->parity, sp->parity == PARITY_NORMAL ? "normal" : (sp->parity == PARITY_FLIP ? "flip" : "both"));
    if (sp->use_radec) {
        double ra,dec,rad;
        xyzarr2radecdeg(sp->centerxyz, &ra, &dec);
        rad = distsq2deg(sp->r2);
        logverb("  Use_radec? yes, (%g, %g), radius %g deg\n", ra, dec, rad);
    } else {
        logverb("  Use_radec? no\n");
    }
    logverb("  Pixel xscale: %g\n", sp->pixel_xscale);
    logverb("  Verify_pix: %g\n", sp->verify_pix);
    logverb("  Code tol: %g\n", sp->codetol);
    logverb("  Dist from quad bonus: %s\n", sp->distance_from_quad_bonus ? "yes" : "no");
    logverb("  Distractor ratio: %g\n", sp->distractor_ratio);
    logverb("  Log tune-up threshold: %g\n", sp->logratio_totune);
    logverb("  Log bail threshold: %g\n", sp->logratio_bail_threshold);
    logverb("  Log stoplooking threshold: %g\n", sp->logratio_stoplooking);
    logverb("  Maxquads %i\n", sp->maxquads);
    logverb("  Maxmatches %i\n", sp->maxmatches);
    logverb("  Set CRPIX? %s", sp->set_crpix ? "yes" : "no\n");
    if (sp->set_crpix) {
        if (sp->set_crpix_center)
            logverb(", center\n");
        else
            logverb(", %g, %g\n", sp->crpix[0], sp->crpix[1]);
    }
    logverb("  Tweak? %s\n", sp->do_tweak ? "yes" : "no");
    if (sp->do_tweak) {
        logverb("    Forward order %i\n", sp->tweak_aborder);
        logverb("    Reverse order %i\n", sp->tweak_abporder);
    }
    logverb("  Indexes: %zu\n", pl_size(sp->indexes));
    for (i=0; i<pl_size(sp->indexes); i++) {
        index_t* ind = pl_get(sp->indexes, i);
        logverb("    %s\n", ind->indexname);
    }
    if (sp->fieldxy) {
      logverb("  Field (processed): %i stars\n", starxy_n(sp->fieldxy));
      for (i=0; i<starxy_n(sp->fieldxy); i++) {
        debug("    xy (%.1f, %.1f), flux %.1f\n",
              starxy_getx(sp->fieldxy, i), starxy_gety(sp->fieldxy, i),
              sp->fieldxy->flux ? starxy_get_flux(sp->fieldxy, i) : 0.0);
      }
    }
    if (sp->fieldxy_orig) {
      logverb("  Field (orig): %i stars\n", starxy_n(sp->fieldxy_orig));
      for (i=0; i<starxy_n(sp->fieldxy_orig); i++) {
        debug("    xy (%.1f, %.1f), flux %.1f\n",
              starxy_getx(sp->fieldxy_orig, i), starxy_gety(sp->fieldxy_orig, i),
              sp->fieldxy_orig->flux ? starxy_get_flux(sp->fieldxy_orig, i) : 0.0);
      }
    }
}


void solver_print_to(const solver_t* sp, FILE* stream) {
    //int oldlevel = log_get_level();
    FILE* oldfid = log_get_fid();
    //log_set_level(LOG_ALL);
    log_to(stream);
    solver_log_params(sp);
    //log_set_level(oldlevel);
    log_to(oldfid);
}

/*
 static MatchObj* matchobj_copy_deep(const MatchObj* mo, MatchObj* dest) {
 if (!dest)
 dest = malloc(sizeof(MatchObj));
 memcpy(dest, mo, sizeof(MatchObj));
 // various modules add things to a mo...
 onefield_matchobj_deep_copy(mo, dest);
 verify_matchobj_deep_copy(mo, dest);
 return dest;
 }
 
 static void matchobj_free_data(MatchObj* mo) {
 verify_free_matchobj(mo);
 onefield_free_matchobj(mo);
 }
 */

static const int A = 0, B = 1, C = 2, D = 3;

// Number of stars in the "backbone" of the quad: stars A and B.
static const int NBACK = 2;

static void find_field_boundaries(solver_t* solver);

static inline double getx(const double* d, int ind) {
    return d[ind*2];
}
static inline double gety(const double* d, int ind) {
    return d[ind*2 + 1];
}
static inline void setx(double* d, int ind, double val) {
    d[ind*2] = val;
}
static inline void sety(double* d, int ind, double val) {
    d[ind*2 + 1] = val;
}

static void field_getxy(solver_t* sp, int index, double* x, double* y) {
    *x = starxy_getx(sp->fieldxy, index);
    *y = starxy_gety(sp->fieldxy, index);
}

static double field_getx(solver_t* sp, int index) {
    return starxy_getx(sp->fieldxy, index);
}
static double field_gety(solver_t* sp, int index) {
    return starxy_gety(sp->fieldxy, index);
}

static void update_timeused(solver_t* sp) {
    double usertime, systime;
    get_resource_stats(&usertime, &systime, NULL);
    sp->timeused = (usertime + systime) - sp->starttime;
    if (sp->timeused < 0.0)
        sp->timeused = 0.0;
}

static void set_matchobj_template(solver_t* solver, MatchObj* mo) {
    if (solver->mo_template)
        memcpy(mo, solver->mo_template, sizeof(MatchObj));
    else
        memset(mo, 0, sizeof(MatchObj));
}

static void get_field_center(solver_t* s, double* cx, double* cy) {
    *cx = 0.5 * (s->field_minx + s->field_maxx);
    *cy = 0.5 * (s->field_miny + s->field_maxy);
}

static void get_field_ll_corner(solver_t* s, double* lx, double* ly) {
    *lx = s->field_minx;
    *ly = s->field_miny;
}

void solver_reset_counters(solver_t* s) {
    s->quit_now = FALSE;
    s->have_best_match = FALSE;
    s->best_match_solves = FALSE;
    s->numtries = 0;
    s->nummatches = 0;
    s->numscaleok = 0;
    s->last_examined_object = 0;
    s->num_cxdx_skipped = 0;
    s->num_radec_skipped = 0;
    s->num_abscale_skipped = 0;
    s->num_verified = 0;
}

double solver_field_width(const solver_t* s) {
    return s->field_maxx - s->field_minx;
}
double solver_field_height(const solver_t* s) {
    return s->field_maxy - s->field_miny;
}

void solver_set_radec(solver_t* s, double ra, double dec, double radius_deg) {
    s->use_radec = TRUE;
    radecdeg2xyzarr(ra, dec, s->centerxyz);
    s->r2 = deg2distsq(radius_deg);
}

void solver_clear_radec(solver_t* s) {
    s->use_radec = FALSE;
}

static void set_center_and_radius(solver_t* solver, MatchObj* mo,
                                  tan_t* tan, sip_t* sip) {
    double cx, cy, lx, ly;
    double xyz[3];
    get_field_center(solver, &cx, &cy);
    get_field_ll_corner(solver, &lx, &ly);
    if (sip) {
        sip_pixelxy2xyzarr(sip, cx, cy, mo->center);
        sip_pixelxy2xyzarr(sip, lx, ly, xyz);
    } else {
        tan_pixelxy2xyzarr(tan, cx, cy, mo->center);
        tan_pixelxy2xyzarr(tan, lx, ly, xyz);
    }
    mo->radius = sqrt(distsq(mo->center, xyz, 3));
    mo->radius_deg = dist2deg(mo->radius);
}

static void set_index(solver_t* s, index_t* index) {
    s->index = index;
    s->rel_index_noise2 = square(index->index_jitter / index->index_scale_lower);
}

static void set_diag(solver_t* s) {
    s->field_diag = hypot(solver_field_width(s), solver_field_height(s));
}

void solver_set_field(solver_t* s, starxy_t* field) {
    solver_free_field(s);
    s->fieldxy_orig = field;
    // Preprocessing happens in "solver_preprocess_field()".
}

void solver_set_field_bounds(solver_t* s, double xlo, double xhi, double ylo, double yhi) {
    s->field_minx = xlo;
    s->field_maxx = xhi;
    s->field_miny = ylo;
    s->field_maxy = yhi;
    set_diag(s);
}

void solver_cleanup_field(solver_t* solver) {
    solver_reset_best_match(solver);
    solver_free_field(solver);
    solver->fieldxy = NULL;
    solver_reset_counters(solver);
}

void solver_verify_sip_wcs(solver_t* solver, sip_t* sip) { //, MatchObj* pmo) {
    int i, nindexes;
    MatchObj mo;
    MatchObj* pmo;
    anbool olddqb;

    pmo = &mo;

    if (!solver->vf)
        solver_preprocess_field(solver);

    // fabricate a match and inject it into the solver.
    set_matchobj_template(solver, pmo);
    memcpy(&(mo.wcstan), &(sip->wcstan), sizeof(tan_t));
    mo.wcs_valid = TRUE;
    mo.scale = sip_pixel_scale(sip);
    set_center_and_radius(solver, pmo, NULL, sip);
    olddqb = solver->distance_from_quad_bonus;
    solver->distance_from_quad_bonus = FALSE;

    nindexes = pl_size(solver->indexes);
    for (i=0; i<nindexes; i++) {
        index_t* index = pl_get(solver->indexes, i);
        set_index(solver, index);
        solver_inject_match(solver, pmo, sip);
    }

    // revert
    solver->distance_from_quad_bonus = olddqb;
}

void solver_add_index(solver_t* solver, index_t* index) {
    pl_append(solver->indexes, index);
}

int solver_n_indices(const solver_t* solver) {
    return pl_size(solver->indexes);
}

index_t* solver_get_index(const solver_t* solver, int i) {
    return pl_get(solver->indexes, i);
}

void solver_reset_best_match(solver_t* sp) {
    // we don't really care about very bad best matches...
    sp->best_logodds = 0;
    memset(&(sp->best_match), 0, sizeof(MatchObj));
    sp->best_index = NULL;
    sp->best_match_solves = FALSE;
    sp->have_best_match = FALSE;
}

void solver_compute_quad_range(const solver_t* sp, const index_t* index,
                               double* minAB, double* maxAB) {
    double scalefudge; // in pixels

    // compute fudge factor for quad scale: what are the extreme
    // ranges of quad scales that should be accepted, given the
    // code tolerance?
    // -what is the maximum number of pixels a C or D star can move
    //  to singlehandedly exceed the code tolerance?
    // -largest quad
    // -smallest arcsec-per-pixel scale

    // -index_scale_upper * 1/sqrt(2) is the side length of
    //  the unit-square of code space, in arcseconds.
    // -that times the code tolerance is how far a C/D star
    //  can move before exceeding the code tolerance, in arcsec.
    // -that divided by the smallest arcsec-per-pixel scale
    //  gives the largest motion in pixels.

    //logverb("Index scale %f, %f\n", 
    //index->index_scale_upper, index->index_scale_lower);
            
    scalefudge = index->index_scale_upper * M_SQRT1_2 *
        sp->codetol / sp->funits_upper;

    if (sp->funits_upper != 0.0) {
        *minAB = index->index_scale_lower / sp->funits_upper;
        *minAB -= scalefudge;
    }
    if (sp->funits_lower != 0.0) {
        *maxAB = index->index_scale_upper / sp->funits_lower;
        *maxAB += scalefudge;
    }
}

static void try_all_codes(const pquad* pq,
                          const int* fieldstars, int dimquad,
                          solver_t* solver, double tol2);

static void try_all_codes_2(const int* fieldstars, int dimquad,
                            const double* code, solver_t* solver,
                            anbool current_parity, double tol2);

static void try_permutations(const int* origstars, int dimquad,
                             const double* origcode,
                             solver_t* solver, anbool current_parity,
                             double tol2,
                             int* stars, double* code,
                             int slot, anbool* placed,
                             kdtree_qres_t** presult);

static void resolve_matches(kdtree_qres_t* krez, const double *field,
                            const int* fstars, int dimquads,
                            solver_t* solver, anbool current_parity);

static int solver_handle_hit(solver_t* sp, MatchObj* mo, sip_t* sip, anbool fake_match);

static void check_scale(pquad* pq, solver_t* s) {
    double dx, dy;
    dx = field_getx(s, pq->fieldB) - field_getx(s, pq->fieldA);
    dy = field_gety(s, pq->fieldB) - field_gety(s, pq->fieldA);
    pq->scale = dx*dx + dy*dy;
    if ((pq->scale < s->minminAB2) ||
        (pq->scale > s->maxmaxAB2)) {
        pq->scale_ok = FALSE;
        return;
    }
    pq->costheta = (dy + dx) / pq->scale;
    pq->sintheta = (dy - dx) / pq->scale;
    pq->rel_field_noise2 = (s->verify_pix * s->verify_pix) / pq->scale;
    pq->scale_ok = TRUE;
}

static void check_inbox(pquad* pq, int start, solver_t* solver) {
    int i;
    double Ax, Ay;
    field_getxy(solver, pq->fieldA, &Ax, &Ay);
    // check which C, D points are inside the circle.
    for (i = start; i < pq->ninbox; i++) {
        double r;
        double Cx, Cy, xxtmp;
        double tol = solver->codetol;
        if (!pq->inbox[i])
            continue;
        field_getxy(solver, i, &Cx, &Cy);
        Cx -= Ax;
        Cy -= Ay;
        xxtmp = Cx;
        Cx = Cx * pq->costheta + Cy * pq->sintheta;
        Cy = -xxtmp * pq->sintheta + Cy * pq->costheta;

        // make sure it's in the circle centered at (0.5, 0.5)
        // with radius 1/sqrt(2) (plus codetol for fudge):
        // (x-1/2)^2 + (y-1/2)^2   <=   (r + codetol)^2
        // x^2-x+1/4 + y^2-y+1/4   <=   (1/sqrt(2) + codetol)^2
        // x^2-x + y^2-y + 1/2     <=   1/2 + sqrt(2)*codetol + codetol^2
        // x^2-x + y^2-y           <=   sqrt(2)*codetol + codetol^2
        r = (Cx * Cx - Cx) + (Cy * Cy - Cy);
        if (r > (tol * (M_SQRT2 + tol))) {
            pq->inbox[i] = FALSE;
            continue;
        }
        setx(pq->xy, i, Cx);
        sety(pq->xy, i, Cy);
    }
}

#if defined DEBUGSOLVER
static void print_inbox(pquad* pq) {
    int i;
    debug("[ ");
    for (i = 0; i < pq->ninbox; i++) {
        if (pq->inbox[i])
            debug("%i ", i);
    }
    debug("] (n %i)\n", pq->ninbox);
}
#else
static void print_inbox(pquad* pq) {}
#endif


void solver_reset_field_size(solver_t* s) {
    s->field_minx = s->field_maxx = s->field_miny = s->field_maxy = 0;
    s->field_diag = 0.0;
}

static void find_field_boundaries(solver_t* solver) {
    // If the bounds haven't been set, use the bounding box.
    if ((solver->field_minx == solver->field_maxx) ||
        (solver->field_miny == solver->field_maxy)) {
        int i;
        solver->field_minx = solver->field_miny =  LARGE_VAL;
        solver->field_maxx = solver->field_maxy = -LARGE_VAL;
        for (i = 0; i < starxy_n(solver->fieldxy); i++) {
            solver->field_minx = MIN(solver->field_minx, field_getx(solver, i));
            solver->field_maxx = MAX(solver->field_maxx, field_getx(solver, i));
            solver->field_miny = MIN(solver->field_miny, field_gety(solver, i));
            solver->field_maxy = MAX(solver->field_maxy, field_gety(solver, i));
        }
    }
    set_diag(solver);
}

void solver_preprocess_field(solver_t* solver) {
    int i;

    // Make a copy of the original x,y list.
    solver->fieldxy = starxy_copy(solver->fieldxy_orig);

    if ((solver->pixel_xscale > 0) && solver->predistort) {
        logerr("Error, can't do both pixel_xscale and predistortion at the same time!");
    }
    if (solver->pixel_xscale > 0) {
        logverb("Applying x-factor of %f to %i stars\n",
                solver->pixel_xscale, starxy_n(solver->fieldxy_orig));
        for (i=0; i<starxy_n(solver->fieldxy); i++)
            solver->fieldxy->x[i] *= solver->pixel_xscale;
    } else if (solver->predistort) {
        logverb("Applying undistortion to %i stars\n", starxy_n(solver->fieldxy_orig));
        // Apply the *un*distortion
        for (i=0; i<starxy_n(solver->fieldxy); i++) {
            double dx, dy;
            sip_pixel_undistortion(solver->predistort,
                                   solver->fieldxy->x[i], solver->fieldxy->y[i],
                                   &dx, &dy);
            solver->fieldxy->x[i] = dx;
            solver->fieldxy->y[i] = dy;
        }
    }

    find_field_boundaries(solver);
    // precompute a kdtree over the field
    solver->vf = verify_field_preprocess(solver->fieldxy);

    solver->vf->do_uniformize = solver->verify_uniformize;
    solver->vf->do_dedup = solver->verify_dedup;

    if (solver->set_crpix && solver->set_crpix_center) {
        solver->crpix[0] = wcs_pixel_center_for_size(solver_field_width(solver));
        solver->crpix[1] = wcs_pixel_center_for_size(solver_field_height(solver));
        logverb("Setting CRPIX to center (%.1f, %.1f) based on image size %i x %i\n",
                solver->crpix[0], solver->crpix[1],
                (int)solver_field_width(solver), (int)solver_field_height(solver));
    }
}

void solver_free_field(solver_t* solver) {
    if (solver->fieldxy)
        starxy_free(solver->fieldxy);
    solver->fieldxy = NULL;
    if (solver->fieldxy_orig)
        starxy_free(solver->fieldxy_orig);
    solver->fieldxy_orig = NULL;
    if (solver->vf)
        verify_field_free(solver->vf);
    solver->vf = NULL;
}

starxy_t* solver_get_field(solver_t* solver) {
    return solver->fieldxy;
}

static double get_tolerance(solver_t* solver) {
    return square(solver->codetol);
    /*
     double maxtol2 = square(solver->codetol);
     double tol2;
     tol2 = 49.0 * (solver->rel_field_noise2 + solver->rel_index_noise2);
     //printf("code tolerance %g.\n", sqrt(tol2));
     if (tol2 > maxtol2)
     tol2 = maxtol2;
     return tol2;
     */
}

/*
 A somewhat tricky recursive function: stars A and B have already been
 chosen, so the code coordinate system has been fixed, and we've
 already determined which other stars will create valid codes (ie, are
 in the "box").  Now we want to build features using all sets of valid
 stars (without permutations).

 pq - data associated with the AB pair.
 field - the array of field star numbers
 fieldoffset - offset into the field array where we should add the first star
 n_to_add - number of stars to add
 adding - the star we're currently adding; in [0, n_to_add).
 fieldtop - the maximum field star number to build quads out of.
 dimquad, solver, tol2 - passed to try_all_codes.
 */
static void add_stars(const pquad* pq, int* field, int fieldoffset,
                      int n_to_add, int adding, int fieldtop,
                      int dimquad,
                      solver_t* solver, double tol2) {
    int bottom;
    int* f = field + fieldoffset;
    // When we're adding the first star, we start from index zero.
    // When we're adding subsequent stars, we start from the previous value
    // plus one, to avoid adding permutations.
    bottom = (adding ? f[adding-1] + 1 : 0);

    // It looks funny that we're using f[adding] as a loop variable, but
    // it's required because try_all_codes needs to know which field stars
    // were used to create the quad (which are stored in the "f" array)
    for (f[adding]=bottom; f[adding]<fieldtop; f[adding]++) {
        if (!pq->inbox[f[adding]])
            continue;
        if (unlikely(solver->quit_now))
            return;

        // If we've hit the end of the recursion (we're adding the last star),
        // call try_all_codes to try the quad we've built.
        if (adding == n_to_add-1) {
            // (when not testing, TRY_ALL_CODES is just try_all_codes.)
            TRY_ALL_CODES(pq, field, dimquad, solver, tol2);
        } else {
            // Else recurse.
            add_stars(pq, field, fieldoffset, n_to_add, adding+1,
                      fieldtop, dimquad, solver, tol2);
        }
    }
}


// The real deal
void solver_run(solver_t* solver) {
    int numxy, newpoint;
    double usertime, systime;
    // first timer callback is called after 1 second
    time_t next_timer_callback_time = time(NULL) + 1;
    pquad* pquads;
    size_t i, num_indexes;
    double tol2;
    int field[DQMAX];

    get_resource_stats(&usertime, &systime, NULL);

    if (!solver->vf)
        solver_preprocess_field(solver);

    memset(field, 0, sizeof(field));

    solver->starttime = usertime + systime;

    numxy = starxy_n(solver->fieldxy);
    if (solver->endobj && (numxy > solver->endobj))
        numxy = solver->endobj;
    if (solver->startobj >= numxy)
        return;
    if (numxy >= 1000) {
        logverb("Limiting search to first 1000 objects\n");
        numxy = 1000;
    }

    num_indexes = pl_size(solver->indexes);
    {
        double minAB2s[num_indexes];
        double maxAB2s[num_indexes];
        solver->minminAB2 = LARGE_VAL;
        solver->maxmaxAB2 = -LARGE_VAL;
        for (i = 0; i < num_indexes; i++) {
            double minAB=0, maxAB=0;
            index_t* index = pl_get(solver->indexes, i);
            // The limits on the size of quads that we try to match, in pixels.
            // Derived from index_scale_* and funits_*.
            solver_compute_quad_range(solver, index, &minAB, &maxAB);
            //logverb("Index \"%s\" quad range %f to %f\n", index->indexname,
            //minAB, maxAB);
            minAB2s[i] = square(minAB);
            maxAB2s[i] = square(maxAB);
            solver->minminAB2 = MIN(solver->minminAB2, minAB2s[i]);
            solver->maxmaxAB2 = MAX(solver->maxmaxAB2, maxAB2s[i]);

            if (index->cx_less_than_dx) {
                solver->cxdx_margin = 1.5 * solver->codetol;
                // FIXME die horribly if the indexes have differing cx_less_than_dx
            }
        }
        solver->minminAB2 = MAX(solver->minminAB2, square(solver->quadsize_min));
        if (solver->quadsize_max != 0.0)
            solver->maxmaxAB2 = MIN(solver->maxmaxAB2, square(solver->quadsize_max));
        logverb("Quad scale range: [%g, %g] pixels\n", sqrt(solver->minminAB2), sqrt(solver->maxmaxAB2));

        // quick-n-dirty scale estimate using stars A,B.
        solver->abscale_high = square(arcsec2rad(solver->funits_upper) * (1.0 + solver->codetol));
        solver->abscale_low  = square(arcsec2rad(solver->funits_lower) * (1.0 - solver->codetol));

        /** Ugh, I want to avoid doing distsq2rad when checking scale,
         but that means correcting for the difference between the
         distance along the curve of the sphere vs the chord distance.
         This affects the lower bound for the largest quads in a messy way...
         This below isn't right.
         solver->abscale_high = square(arcsec2rad(solver->funits_upper) * (1.0 + solver->codetol));
         solver->abscale_low = arcsec2rad(solver->funits_lower) * (1.0 - solver->codetol) *
         MIN(M_PI, arcsec2rad(field_diag * solver->funits_upper)) ...
         */

        pquads = calloc((size_t)numxy * (size_t)numxy, sizeof(pquad));

        /* We maintain an array of "potential quads" (pquad) structs, where
         * each struct corresponds to one choice of stars A and B; the struct
         * at index (B * numxy + A) holds information about quads that could be
         * created using stars A,B.
         *
         * (We only use the above-diagonal elements of this 2D array because
         * A<B.)
         *
         * For each AB pair, we cache the scale and the rotation parameters,
         * and we keep an array "inbox" of length "numxy" of booleans, one for
         * each star, which say whether that star is eligible to be star C or D
         * of a quad with AB at the corners.  (Obviously A and B aren't
         * eligible).
         *
         * The "ninbox" parameter is somewhat misnamed - it says that "inbox"
         * elements in the range [0, ninbox) have been initialized.
         */

        /* (See explanatory paragraph below) If "solver->startobj" isn't zero,
         * then we need to initialize the triangle of "pquads" up to
         * A=startobj-2, B=startobj-1. */
        if (solver->startobj) {
            debug("startobj > 0; priming pquad arrays.\n");
            for (field[B] = 0; field[B] < solver->startobj; field[B]++) {
                for (field[A] = 0; field[A] < field[B]; field[A]++) {
                    pquad* pq = pquads + field[B] * numxy + field[A];
                    pq->fieldA = field[A];
                    pq->fieldB = field[B];
                    debug("trying A=%i, B=%i\n", field[A], field[B]);
                    check_scale(pq, solver);
                    if (!pq->scale_ok) {
                        debug("  bad scale for A=%i, B=%i\n", field[A], field[B]);
                        continue;
                    }
                    pq->xy = malloc(numxy * 2 * sizeof(double));
                    pq->inbox = malloc(numxy * sizeof(anbool));
                    memset(pq->inbox, TRUE, solver->startobj);
                    pq->ninbox = solver->startobj;
                    pq->inbox[field[A]] = FALSE;
                    pq->inbox[field[B]] = FALSE;
                    check_inbox(pq, 0, solver);
                    debug("  inbox(A=%i, B=%i): ", field[A], field[B]);
                    print_inbox(pq);
                }
            }
        }

        /* Each time through the "for" loop below, we consider a new star
         * ("newpoint").  First, we try building all quads that have the new
         * star on the diagonal (star B).  Then, we try building all quads that
         * have the star not on the diagonal (star D).
         * 
         * For each AB pair, we have a "potential_quad" or "pquad" struct.
         * This caches the computation we need to do: deciding whether the
         * scale is acceptable, computing the transformation to code
         * coordinates, and deciding which C,D stars are in the circle.
         */
        for (newpoint = solver->startobj; newpoint < numxy; newpoint++) {

            debug("Trying newpoint=%i (%.1f,%.1f)\n", newpoint,
                  field_getx(solver,newpoint), field_gety(solver,newpoint));

            // Give our caller a chance to cancel us midway. The callback
            // returns how long to wait before calling again.

            if (solver->timer_callback) {
                time_t delay;
                time_t now = time(NULL);
                if (now > next_timer_callback_time) {
                    update_timeused(solver);
                    delay = solver->timer_callback(solver->userdata);
                    if (delay == 0) // Canceled
                        break;
                    next_timer_callback_time = now + delay;
                }
            }

            solver->last_examined_object = newpoint;
            // quads with the new star on the diagonal:
            field[B] = newpoint;
            debug("Trying quads with B=%i\n", newpoint);
	
            // first do an index-independent scale check...
            for (field[A] = 0; field[A] < newpoint; field[A]++) {
                // initialize the "pquad" struct for this AB combo.
                pquad* pq = pquads + field[B] * numxy + field[A];
                pq->fieldA = field[A];
                pq->fieldB = field[B];
                debug("  trying A=%i, B=%i\n", field[A], field[B]);
                check_scale(pq, solver);
                if (!pq->scale_ok) {
                    debug("    bad scale for A=%i, B=%i\n", field[A], field[B]);
                    continue;
                }
                // initialize the "inbox" array:
                pq->inbox = malloc(numxy * sizeof(anbool));
                pq->xy = malloc(numxy * 2 * sizeof(double));
                // -try all stars up to "newpoint"...
                assert(sizeof(anbool) == 1);
                memset(pq->inbox, TRUE, newpoint + 1);
                pq->ninbox = newpoint + 1;
                // -except A and B.
                pq->inbox[field[A]] = FALSE;
                pq->inbox[field[B]] = FALSE;
                check_inbox(pq, 0, solver);
                debug("    inbox(A=%i, B=%i): ", field[A], field[B]);
                print_inbox(pq);
            }

            // Now iterate through the different indices
            for (i = 0; i < num_indexes; i++) {
                index_t* index = pl_get(solver->indexes, i);
                int dimquads;
                set_index(solver, index);
                dimquads = index_dimquads(index);
                for (field[A] = 0; field[A] < newpoint; field[A]++) {
                    // initialize the "pquad" struct for this AB combo.
                    pquad* pq = pquads + field[B] * numxy + field[A];
                    if (!pq->scale_ok)
                        continue;
                    if ((pq->scale < minAB2s[i]) ||
                        (pq->scale > maxAB2s[i]))
                        continue;
                    // set code tolerance for this index and AB pair...
                    solver->rel_field_noise2 = pq->rel_field_noise2;
                    tol2 = get_tolerance(solver);
                    // Now look at all sets of (C, D, ...) stars (subject to field[C] < field[D] < ...)
                    // ("dimquads - 2" because we've set stars A and B at this point)
                    add_stars(pq, field, C, dimquads-2, 0, newpoint, dimquads, solver, tol2);
                    if (solver->quit_now)
                        goto quitnow;
                }
            }

            if (solver->quit_now)
                goto quitnow;

            // Now try building quads with the new star not on the diagonal:
            field[C] = newpoint;
            // (in this loop field[C] > field[D])
            debug("Trying quads with C=%i\n", newpoint);
            for (field[A] = 0; field[A] < newpoint; field[A]++) {
                for (field[B] = field[A] + 1; field[B] < newpoint; field[B]++) {
                    // grab the "pquad" for this AB combo
                    pquad* pq = pquads + field[B] * numxy + field[A];
                    if (!pq->scale_ok) {
                        debug("  bad scale for A=%i, B=%i\n", field[A], field[B]);
                        continue;
                    }
                    // test if this C is in the box:
                    pq->inbox[field[C]] = TRUE;
                    pq->ninbox = field[C] + 1;
                    check_inbox(pq, field[C], solver);
                    if (!pq->inbox[field[C]]) {
                        debug("  C is not in the box for A=%i, B=%i\n", field[A], field[B]);
                        continue;
                    }
                    debug("  C is in the box for A=%i, B=%i\n", field[A], field[B]);
                    debug("    box now:");
                    print_inbox(pq);
                    debug("\n");

                    solver->rel_field_noise2 = pq->rel_field_noise2;

                    for (i = 0; i < pl_size(solver->indexes); i++) {
                        int dimquads;
                        index_t* index = pl_get(solver->indexes, i);
                        if ((pq->scale < minAB2s[i]) ||
                            (pq->scale > maxAB2s[i]))
                            continue;
                        set_index(solver, index);
                        dimquads = index_dimquads(index);

                        tol2 = get_tolerance(solver);

                        if (dimquads > 3) {
                            // ("dimquads - 3" because we've set stars A, B, and C at this point)
                            add_stars(pq, field, D, dimquads-3, 0, newpoint, dimquads, solver, tol2);
                        } else {
                            TRY_ALL_CODES(pq, field, dimquads, solver, tol2);
                        }
                        if (solver->quit_now)
                            goto quitnow;
                    }
                }
            }
            logverb("object %u of %u: %i quads tried, %i matched.\n",
                    newpoint + 1, numxy, solver->numtries, solver->nummatches);

            if ((solver->maxquads && (solver->numtries >= solver->maxquads))
                || (solver->maxmatches && (solver->nummatches >= solver->maxmatches))
                || solver->quit_now)
                break;
        }

    quitnow:
        for (i = 0; i < (numxy*numxy); i++) {
            pquad* pq = pquads + i;
            free(pq->inbox);
            free(pq->xy);
        }
        free(pquads);
    }
}

/**
 All the stars in this quad have been chosen.  Figure out which
 permutations of stars CDE are valid and search for matches.
 */
static void try_all_codes(const pquad* pq,
                          const int* fieldstars, int dimquad,
                          solver_t* solver, double tol2) {
    int dimcode = (dimquad - 2) * 2;
    double code[DCMAX];
    double flipcode[DCMAX];
    int i;

    solver->numtries++;

    debug("  trying quad [");
    for (i=0; i<dimquad; i++) {
        debug("%s%i", (i?" ":""), fieldstars[i]);
    }
    debug("]\n");

    for (i=0; i<dimquad-NBACK; i++) {
        code[2*i  ] = getx(pq->xy, fieldstars[NBACK+i]);
        code[2*i+1] = gety(pq->xy, fieldstars[NBACK+i]);
    }

    if (solver->parity == PARITY_NORMAL ||
        solver->parity == PARITY_BOTH) {

        debug("    trying normal parity: code=[");
        for (i=0; i<dimcode; i++)
            debug("%s%g", (i?", ":""), code[i]);
        debug("].\n");

        try_all_codes_2(fieldstars, dimquad, code, solver, FALSE, tol2);
    }
    if (solver->parity == PARITY_FLIP ||
        solver->parity == PARITY_BOTH) {

        quad_flip_parity(code, flipcode, dimcode);

        debug("    trying reverse parity: code=[");
        for (i=0; i<dimcode; i++)
            debug("%s%g", (i?", ":""), flipcode[i]);
        debug("].\n");

        try_all_codes_2(fieldstars, dimquad, flipcode, solver, TRUE, tol2);
    }
}

/**
 This function tries the quad with the "backbone" stars A and B in
 normal and flipped configurations.
 */
static void try_all_codes_2(const int* fieldstars, int dimquad,
                            const double* code, solver_t* solver,
                            anbool current_parity, double tol2) {
    int i;
    kdtree_qres_t* result = NULL;
    int dimcode = (dimquad - 2) * 2;
    int stars[DQMAX];
    double flipcode[DCMAX];

    // We actually only use elements up to dimquads-2.
    anbool placed[DQMAX];

    // Un-flipped:
    stars[0] = fieldstars[0];
    stars[1] = fieldstars[1];

    for (i=0; i<DQMAX; i++)
        placed[i] = FALSE;

    try_permutations(fieldstars, dimquad, code, solver, current_parity,
                     tol2, stars, NULL, 0, placed, &result);
    if (unlikely(solver->quit_now))
        goto bailout;

    // Flipped:
    stars[0] = fieldstars[1];
    stars[1] = fieldstars[0];

    for (i=0; i<dimcode; i++)
        flipcode[i] = 1.0 - code[i];

    for (i=0; i<DQMAX; i++)
        placed[i] = FALSE;

    try_permutations(fieldstars, dimquad, flipcode, solver, current_parity,
                     tol2, stars, NULL, 0, placed, &result);

 bailout:
    kdtree_free_query(result);
}

/**
 This functions tries different permutations of the non-backbone
 stars C [, D [,E ] ]
 */
static void try_permutations(const int* origstars, int dimquad,
                             const double* origcode,
                             solver_t* solver, anbool current_parity,
                             double tol2,
                             int* stars, double* code,
                             int slot, anbool* placed,
                             kdtree_qres_t** presult) {
    int i;
    int options = KD_OPTIONS_SMALL_RADIUS | KD_OPTIONS_COMPUTE_DISTS |
        KD_OPTIONS_NO_RESIZE_RESULTS | KD_OPTIONS_USE_SPLIT;
    double mycode[DCMAX];
    int Nstars = dimquad - NBACK;
    int lastslot = dimquad - NBACK - 1;
    /*
     This is a recursive function that tries all combinations of the
     "internal" stars (ie, not stars A,B that form the "backbone" of
     the quad).

     We fill the "stars" array with the star IDs (from "origstars") of
     the stars that form the quad, while simultaneously filling the
     "code" array with the corresponding code coordinates (from
     "origcode").

     For example, if "dimquad" is 5, and "origstars" contains
     A,B,C,D,E, we want to call "resolve_matches" with the following
     combinations in "stars":

     AB CDE
     AB CED
     AB DCE
     AB DEC
     AB ECD
     AB EDC

     This call will try to put each star in "slot" in turn, then for
     each one recurse to "slot" in the rest of the stars.

     Note that we are filling stars[2], stars[3], etc; the first two
     elements are already filled by stars A and B.
     */

    if (code == NULL)
        code = mycode;

    // We try putting each star that hasn't already been placed in
    // this "slot".
    for (i=0; i<Nstars; i++) {
        if (placed[i])
            continue;

        // Check cx <= dx, if we're a "dx".
        if (slot > 0 && solver->index->cx_less_than_dx) {
            if (code[2*(slot - 1) +0] > origcode[2*i +0] + solver->cxdx_margin) {
                debug("cx <= dx check failed: %g > %g + %g\n",
                      code[2*(slot - 1) +0], origcode[2*i +0],
                      solver->cxdx_margin);
                solver->num_cxdx_skipped++;
                continue;
            }
        }

        // Slot in this star...
        stars[slot + NBACK] = origstars[i + NBACK];
        code[2*slot +0] = origcode[2*i +0];
        code[2*slot +1] = origcode[2*i +1];

        // Check meanx <= 1/2.
        if (solver->index->cx_less_than_dx &&
            solver->index->meanx_less_than_half) {
            // Check the "cx + dx <= 1" condition (for quads); in general,
            // combined with the "cx <= dx" condition, this means that the
            // mean(x) <= 1/2.
            int j;
            double meanx = 0;
            for (j=0; j<=slot; j++)
                meanx += code[2*j];
            meanx /= (slot+1);
            if (meanx > 0.5 + solver->cxdx_margin) {
                debug("meanx <= 0.5 check failed: %g > 0.5 + %g\n", 
                      meanx, solver->cxdx_margin);
                solver->num_meanx_skipped++;
                continue;
            }
        }

        // If we have more slots to fill...
        if (slot < lastslot) {
            placed[i] = TRUE;
            try_permutations(origstars, dimquad, origcode, solver,
                             current_parity, tol2, stars, code, 
                             slot+1, placed, presult);
            placed[i] = FALSE;

        } else {
#if defined(TESTING_TRYPERMUTATIONS)
            TEST_TRY_PERMUTATIONS(stars, code, dimquad, solver);
            continue;
#endif
				
            // Search with the code we've built.
            *presult = kdtree_rangesearch_options_reuse
                (solver->index->codekd->tree, *presult, code, tol2, options);
            //debug("      trying ABCD = [%i %i %i %i]: %i results.\n",
            //fstars[A], fstars[B], fstars[C], fstars[D], result->nres);

            if ((*presult)->nres) {
                double pixvals[DQMAX*2];
                int j;
                for (j=0; j<dimquad; j++) {
                    setx(pixvals, j, field_getx(solver, stars[j]));
                    sety(pixvals, j, field_gety(solver, stars[j]));
                }
                resolve_matches(*presult, pixvals, stars, dimquad, solver,
                                current_parity);
            }
            if (unlikely(solver->quit_now))
                return;
        }
    }
}

static void resolve_matches(kdtree_qres_t* krez, const double *field_xy,
                            const int* fieldstars, int dimquads,
                            solver_t* solver, anbool current_parity) {
    // "field_xy" contains the xy pixel coordinates of stars A,B,C,D forming the quad
    //    [x_A,y_A, x_B,y_B, x_C,y_C, ...]
    int jj, thisquadno;
    MatchObj mo;
    unsigned int star[dimquads];

    assert(krez);

    for (jj = 0; jj < krez->nres; jj++) {
        double starxyz[dimquads*3];
        double scale;
        double arcsecperpix;
        tan_t wcs;
        int i;
        anbool outofbounds = FALSE;
        double abscale;

        solver->nummatches++;
        thisquadno = krez->inds[jj];
        quadfile_get_stars(solver->index->quads, thisquadno, star);
        for (i=0; i<dimquads; i++) {
            startree_get(solver->index->starkd, star[i], starxyz + 3*i);
            if (solver->use_radec)
                if (distsq(starxyz + 3*i, solver->centerxyz, 3) > solver->r2) {
                    outofbounds = TRUE;
                    break;
                }
        }
        if (outofbounds) {
            debug("Quad match is out of bounds.\n");
            solver->num_radec_skipped++;
            continue;
        }

        debug("        stars [");
        for (i=0; i<dimquads; i++)
            debug("%s%i", (i?" ":""), star[i]);
        debug("]\n");

        // Quick-n-dirty scale estimate based on two stars.
        // in (rad per pix)**2
        abscale = square(distsq2rad(distsq(starxyz, starxyz+3, 3))) / 
            distsq(field_xy, field_xy+2, 2);
        if (abscale > solver->abscale_high ||
            abscale < solver->abscale_low) {
            solver->num_abscale_skipped++;
            continue;
        }

        // compute TAN projection from the matching quad alone.
        if (fit_tan_wcs(starxyz, field_xy, dimquads, &wcs, &scale)) {
            // bad quad.
            logverb("bad quad at %s:%i\n", __FILE__, __LINE__);
            continue;
        }
        arcsecperpix = scale * 3600.0;

        // FIXME - should there be scale fudge here?
        if (arcsecperpix > solver->funits_upper ||
            arcsecperpix < solver->funits_lower) {
            debug("          bad scale (%g arcsec/pix, range %g %g)\n",
                  arcsecperpix, solver->funits_lower, solver->funits_upper);
            continue;
        }
        solver->numscaleok++;

        set_matchobj_template(solver, &mo);
        memcpy(&(mo.wcstan), &wcs, sizeof(tan_t));
        mo.wcs_valid = TRUE;
        mo.code_err = krez->sdists[jj];
        mo.scale = arcsecperpix;
        mo.parity = current_parity;
        mo.quads_tried = solver->numtries;
        mo.quads_matched = solver->nummatches;
        mo.quads_scaleok = solver->numscaleok;
        mo.quad_npeers = krez->nres;
        mo.timeused = solver->timeused;
        mo.quadno = thisquadno;
        mo.dimquads = dimquads;
        for (i=0; i<dimquads; i++) {
            mo.star[i] = star[i];
            mo.field[i] = fieldstars[i];
            mo.ids[i] = 0;
        }

        memcpy(mo.quadpix, field_xy, 2 * dimquads * sizeof(double));
        memcpy(mo.quadxyz, starxyz, 3 * dimquads * sizeof(double));

        set_center_and_radius(solver, &mo, &(mo.wcstan), NULL);

        if (solver_handle_hit(solver, &mo, NULL, FALSE))
            solver->quit_now = TRUE;

        if (unlikely(solver->quit_now))
            return;
    }
}

void solver_inject_match(solver_t* solver, MatchObj* mo, sip_t* sip) {
    solver_handle_hit(solver, mo, sip, TRUE);
}

static int solver_handle_hit(solver_t* sp, MatchObj* mo, sip_t* verifysip,
                             anbool fake_match) {
    double match_distance_in_pixels2;
    anbool solved;
    double logaccept;

    mo->indexid = sp->index->indexid;
    mo->healpix = sp->index->healpix;
    mo->hpnside = sp->index->hpnside;
    mo->wcstan.imagew = sp->field_maxx;
    mo->wcstan.imageh = sp->field_maxy;
    mo->dimquads = quadfile_dimquads(sp->index->quads);

    match_distance_in_pixels2 = square(sp->verify_pix) +
        square(sp->index->index_jitter / mo->scale);

    logaccept = MIN(sp->logratio_tokeep, sp->logratio_totune);

    verify_hit(sp->index->starkd, sp->index->cutnside,
               mo, verifysip, sp->vf, match_distance_in_pixels2,
               sp->distractor_ratio, sp->field_maxx, sp->field_maxy,
               sp->logratio_bail_threshold, logaccept,
               sp->logratio_stoplooking,
               sp->distance_from_quad_bonus, fake_match);
    mo->nverified = sp->num_verified++;

    if (mo->logodds >= sp->best_logodds) {
        sp->best_logodds = mo->logodds;
        logverb("Got a new best match: logodds %g.\n", mo->logodds);
    }

    if (mo->logodds >= sp->logratio_totune &&
        mo->logodds < sp->logratio_tokeep) {
        logverb("Trying to tune up this solution (logodds = %g; %g)...\n",
                mo->logodds, exp(mo->logodds));
        solver_tweak2(sp, mo, 1, NULL);
        logverb("After tuning, logodds = %g (%g)\n",
                mo->logodds, exp(mo->logodds));

        // Since we tuned up this solution, we can't just accept the
        // resulting log-odds at face value.
        if (!fake_match) {
            verify_hit(sp->index->starkd, sp->index->cutnside,
                       mo, mo->sip, sp->vf, match_distance_in_pixels2,
                       sp->distractor_ratio,
                       sp->field_maxx, sp->field_maxy,
                       sp->logratio_bail_threshold,
                       sp->logratio_tokeep,
                       sp->logratio_stoplooking,
                       sp->distance_from_quad_bonus,
                       fake_match);
            logverb("Checking tuned result: logodds = %g (%g)\n",
                    mo->logodds, exp(mo->logodds));
        }
    }

    if (mo->logodds < sp->logratio_toprint)
        return FALSE;

    // Also copy original field star coordinates
    //mo.quadpix_orig
    logverb("mo field stars:\n");
    int i;
    for (i=0; i<mo->dimquads; i++) {
        logverb("  star %i; field_xy %.1f,%.1f, field_orig %.1f,%.1f\n",
               mo->field[i], mo->quadpix[2*i+0], mo->quadpix[2*i+1],
               starxy_getx(sp->fieldxy_orig, mo->field[i]),
               starxy_gety(sp->fieldxy_orig, mo->field[i]));
        mo->quadpix_orig[2*i+0] = starxy_getx(sp->fieldxy_orig, mo->field[i]);
        mo->quadpix_orig[2*i+1] = starxy_gety(sp->fieldxy_orig, mo->field[i]);
    }
    
    update_timeused(sp);
    mo->timeused = sp->timeused;

    matchobj_print(mo, log_get_level());

    if (mo->logodds < sp->logratio_tokeep)
        return FALSE;

    logverb("Pixel scale: %g arcsec/pix.\n", mo->scale);
    logverb("Parity: %s.\n", (mo->parity ? "neg" : "pos"));

    mo->index = sp->index;
    mo->index_jitter = sp->index->index_jitter;

    if (sp->predistort || (sp->pixel_xscale > 0)) {
        int i;
        double* matchxy;
        double* matchxyz;
        double* weights;
        int N;
        int Ngood;
        double dx,dy;

        // Apply the distortion.
        if (sp->predistort)
            logverb("Applying the distortion pattern and recomputing WCS...\n");
        else
            logverb("Applying pixel scaling and recomputing WCS...\n");

        if (log_get_level() >= LOG_VERB) {
            printf("Initial WCS:\n");
            tan_print(&(mo->wcstan));
        }

        // this includes conflicts and distractors; we won't fill these arrays.
        N = mo->nbest;
        matchxy = malloc(N * 2 * sizeof(double));
        matchxyz = malloc(N * 3 * sizeof(double));
        weights = malloc(N * sizeof(double));

        Ngood = 0;
        for (i=0; i<N; i++) {
            if (mo->theta[i] < 0)
                continue;
            // Plug in the original (distorted) coordinates
            dx = starxy_get_x(sp->fieldxy_orig, i);
            dy = starxy_get_y(sp->fieldxy_orig, i);
            matchxy[2*Ngood + 0] = dx;
            matchxy[2*Ngood + 1] = dy;
            memcpy(matchxyz + 3*Ngood, mo->refxyz + 3*mo->theta[i],
                   3*sizeof(double));
            weights[Ngood] = verify_logodds_to_weight(mo->matchodds[i]);

            double xx,yy;
            Unused anbool ok;
            ok = tan_xyzarr2pixelxy(&mo->wcstan, matchxyz+3*Ngood, &xx, &yy);
            assert(ok);
            logverb("match: ref(%.1f, %.1f) -- undist(%.1f, %.1f) --> dist(%.1f, %.1f)\n",
                    xx, yy, starxy_get_x(sp->fieldxy, i), starxy_get_y(sp->fieldxy, i), dx, dy);
            Ngood++;
        }

        if (sp->do_tweak) {
            // Compute the SIP solution using the correspondences
            // found during verify(), but with the original (distorted) positions.
            sip_t* sip = sip_create();
            memset(sip, 0, sizeof(sip_t));
            memcpy(&(sip->wcstan), &(mo->wcstan), sizeof(tan_t));
            sip->a_order = sip->b_order = sp->tweak_aborder;
            sip->ap_order = sip->bp_order = sp->tweak_abporder;
            sip->wcstan.imagew = solver_field_width(sp);
            sip->wcstan.imageh = solver_field_height(sp);
            if (sp->set_crpix) {
                sip->wcstan.crpix[0] = sp->crpix[0];
                sip->wcstan.crpix[1] = sp->crpix[1];
                if (sp->predistort) {
                    // find matching crval...
                    sip_pixel_undistortion(sp->predistort,
                                           sp->crpix[0], sp->crpix[1], &dx, &dy);
                } else {
                    dx = sp->crpix[0] / sp->pixel_xscale;
                    dy = sp->crpix[1];
                }
                tan_pixelxy2radecarr(&mo->wcstan, dx, dy, sip->wcstan.crval);

            } else {
                // keep TAN WCS's crval but distort the crpix.
                if (sp->predistort) {
                    sip_pixel_distortion(sp->predistort,
                                         mo->wcstan.crpix[0], mo->wcstan.crpix[1],
                                         sip->wcstan.crpix+0, sip->wcstan.crpix+1);
                } else {
                    sip->wcstan.crpix[0] = mo->wcstan.crpix[0] / sp->pixel_xscale;
                    sip->wcstan.crpix[1] = mo->wcstan.crpix[1];
                }
            }

            if (log_get_level() >= LOG_VERB) {
                printf("Initial SIP on distorted positions:\n");
                sip_print(sip);
            }
            
            int doshift = 1;
            fit_sip_wcs(matchxyz, matchxy, weights, Ngood, &(sip->wcstan),
                        sp->tweak_aborder, sp->tweak_abporder, doshift,
                        sip);

            if (log_get_level() >= LOG_VERB) {
                printf("Final SIP on distorted positions:\n");
                sip_print(sip);
            }

            for (i=0; i<Ngood; i++) {
                double xx,yy;
                Unused anbool ok;
                ok = sip_xyzarr2pixelxy(sip, matchxyz+3*i, &xx, &yy);
                assert(ok);
                logverb("match: ref(%.1f, %.1f) -- dist(%.1f, %.1f)\n",
                        xx, yy, matchxy[2*i+0], matchxy[2*i+1]);
            }
            mo->sip = sip;
            
        } else {
            // Take the coordinates after applying the --predistort,
            // fit a TAN to those, and include the --predistort in the output
            // SIP WCS.
            if (sp->predistort) {
                Ngood = 0;
                for (i=0; i<N; i++) {
                    if (mo->theta[i] < 0)
                        continue;
                    // Plug in the (undistorted) coordinates
                    dx = starxy_get_x(sp->fieldxy, i);
                    dy = starxy_get_y(sp->fieldxy, i);
                    matchxy[2*Ngood + 0] = dx;
                    matchxy[2*Ngood + 1] = dy;
                }
            }

            // Compute new TAN WCS...?
            fit_tan_wcs_weighted(matchxyz, matchxy, weights, Ngood,
                                 &mo->wcstan, NULL);
            if (sp->set_crpix) {
                tan_t wcs2;
                fit_tan_wcs_move_tangent_point(matchxyz, matchxy, Ngood,
                                               sp->crpix, &mo->wcstan, &wcs2);
                fit_tan_wcs_move_tangent_point(matchxyz, matchxy, Ngood,
                                               sp->crpix, &wcs2, &mo->wcstan);
            }
            if (sp->predistort) {
                // Copy the distortion
                sip_t* sip = sip_create();
                memcpy(sip, sp->predistort, sizeof(sip_t));
                memcpy(&sip->wcstan, &mo->wcstan, sizeof(tan_t));
                mo->sip = sip;
            }
        }

        free(matchxy);
        free(matchxyz);
        free(weights);

    } else if (sp->do_tweak) {
        solver_tweak2(sp, mo, sp->tweak_aborder, verifysip);

    } else if (!verifysip && sp->set_crpix) {
        tan_t wcs2;
        tan_t wcs3;
        fit_tan_wcs_move_tangent_point(mo->quadxyz, mo->quadpix, mo->dimquads,
                                       sp->crpix, &(mo->wcstan), &wcs2);
        fit_tan_wcs_move_tangent_point(mo->quadxyz, mo->quadpix, mo->dimquads,
                                       sp->crpix, &wcs2, &wcs3);
        memcpy(&(mo->wcstan), &wcs3, sizeof(tan_t));
        /*
         Good test case:
         http://antwrp.gsfc.nasa.gov/apod/image/0912/Geminid2007_pacholka850wp.jpg
         solve-field --config backend.cfg Geminid2007_pacholka850wp.xy \
         --scale-low 10 --scale-units degwidth -v --no-tweak --continue --new-fits none \
         -o 4 --crpix-center --depth 40-45

         printf("Original WCS:\n");
         tan_print_to(&(mo->wcstan), stdout);
         printf("\n");
         printf("Moved WCS:\n");
         tan_print_to(&wcs2, stdout);
         printf("\n");
         printf("Moved again WCS:\n");
         tan_print_to(&wcs3, stdout);
         printf("\n");
         */
    }

    // If the user didn't supply a callback, or if the callback
    // returns TRUE, consider it solved.
    solved = (!sp->record_match_callback ||
              sp->record_match_callback(mo, sp->userdata));

    // New best match?
    if (!sp->have_best_match || (mo->logodds > sp->best_match.logodds)) {
        if (sp->have_best_match)
            verify_free_matchobj(&sp->best_match);
        memcpy(&sp->best_match, mo, sizeof(MatchObj));
        sp->have_best_match = TRUE;
        sp->best_index = sp->index;
    } else {
        verify_free_matchobj(mo);
    }

    if (solved) {
        sp->best_match_solves = TRUE;
        return TRUE;
    }
    return FALSE;
}

solver_t* solver_new() {
    solver_t* solver = calloc(1, sizeof(solver_t));
    solver_set_default_values(solver);
    return solver;
}

void solver_set_default_values(solver_t* solver) {
    memset(solver, 0, sizeof(solver_t));
    solver->indexes = pl_new(16);
    solver->funits_upper = LARGE_VAL;
    solver->logratio_bail_threshold = log(1e-100);
    solver->logratio_stoplooking = LARGE_VAL;
    solver->logratio_totune = LARGE_VAL;
    solver->parity = DEFAULT_PARITY;
    solver->codetol = DEFAULT_CODE_TOL;
    solver->distractor_ratio = DEFAULT_DISTRACTOR_RATIO;
    solver->verify_pix = DEFAULT_VERIFY_PIX;
    solver->verify_uniformize = TRUE;
    solver->verify_dedup = TRUE;
    solver->distance_from_quad_bonus = TRUE;
    solver->tweak_aborder = DEFAULT_TWEAK_ABORDER;
    solver->tweak_abporder = DEFAULT_TWEAK_ABPORDER;
}

void solver_clear_indexes(solver_t* solver) {
    pl_remove_all(solver->indexes);
    solver->index = NULL;
}

void solver_cleanup(solver_t* solver) {
    solver_free_field(solver);
    pl_free(solver->indexes);
    solver->indexes = NULL;
    if (solver->have_best_match) {
        verify_free_matchobj(&solver->best_match);
        solver->have_best_match = FALSE;
    }
    if (solver->predistort)
        sip_free(solver->predistort);
    solver->predistort = NULL;
}

void solver_free(solver_t* solver) {
    if (!solver) return;
    solver_cleanup(solver);
    free(solver);
}
