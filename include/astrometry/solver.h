/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <time.h>

#include "astrometry/starutil.h"
#include "astrometry/starxy.h"
#include "astrometry/kdtree.h"
#include "astrometry/bl.h"
#include "astrometry/matchobj.h"
#include "astrometry/quadfile.h"
#include "astrometry/starkd.h"
#include "astrometry/codekd.h"
#include "astrometry/index.h"
#include "astrometry/verify.h"
#include "astrometry/sip.h"
#include "astrometry/an-bool.h"

enum {
    PARITY_NORMAL,
    PARITY_FLIP,
    PARITY_BOTH
};

#define DEFAULT_CODE_TOL .01
#define DEFAULT_PARITY PARITY_BOTH
#define DEFAULT_TWEAK_ABORDER 3
#define DEFAULT_TWEAK_ABPORDER 3
#define DEFAULT_DISTRACTOR_RATIO 0.25
#define DEFAULT_VERIFY_PIX 1.0
#define DEFAULT_BAIL_THRESHOLD 1e-100

struct verify_field_t;
struct solver_t {

    // FIELDS REQUIRED FROM THE CALLER BEFORE CALLING SOLVER_RUN
    // =========================================================
	
    // The set of indexes.  Caller must add with solver_add_index()
    pl* indexes;

    // The field to solve
    starxy_t* fieldxy;

    // X axis scaling to apply to pixels before solving -- for pixels
    // that view a rectangular chunk of sky
    double pixel_xscale;

    // Distortion pattern to apply before solving.
    sip_t* predistort;
    starxy_t* fieldxy_orig;

    // Limits on the image pixel scale in [arcsec per pixel].
    double funits_lower;
    double funits_upper;

    double logratio_toprint;
    double logratio_tokeep;

    double logratio_totune;

    // Callback; called for each match found whose log-odds ratio is above
    // "logratio_record_threshold".  The second parameter is "userdata".
    anbool (*record_match_callback)(MatchObj*, void*);

    // User data passed to the callbacks
    void* userdata;

    // Assume that stars far from the matched quad will have larger positional
    // variance?
    anbool distance_from_quad_bonus;

    anbool verify_uniformize;
    anbool verify_dedup;

    anbool do_tweak;

    int tweak_aborder;
    int tweak_abporder;


    // OPTIONAL FIELDS WITH SENSIBLE DEFAULTS
    // ======================================

    // The positional noise in the field, in pixels.
    double verify_pix;

    // Fraction of distractors in [0,1].
    double distractor_ratio;

    // Code tolerance in 4D codespace L2 distance.
    double codetol;

    // Minimum size of field quads to try, in pixels.
    double quadsize_min;
    // Maximum size of field quads to try, in pixels.
    double quadsize_max;

    // The first and last field objects to look at; default is all of them.
    int startobj;
    int endobj;

    // One of PARITY_NORMAL, PARITY_FLIP, or PARITY_BOTH.  Are the X and Y axes of
    // the image flipped?  Default PARITY_BOTH.
    int parity;

    // Only accept matches within a radius of a given RA,Dec position?
    anbool use_radec;
    double centerxyz[3];
    double r2;
	
    // During verification, if the log-odds ratio drops to this level, we bail out and
    // assume it's not a match.  Default log(1e-100).
    double logratio_bail_threshold;

    // During verification, if the log-odds ratio rises above this level, we accept the
    // match and bail out.  Default: LARGE_VAL (ie, don't bail out: keep going to find the
    // maximum Bayes factor value).
    double logratio_stoplooking;

    // Number of field quads to try or zero for no limit.
    int maxquads;
    // Number of quad matches to try or zero for no limit.
    int maxmatches;

    // Force CRPIX to be the given point "crpix", or the center of the image?
    anbool set_crpix;
    anbool set_crpix_center;
    double crpix[2];

    // MatchObj template: if non-NULL, whenever a match is found, we first memcpy()
    // this template, then set the fields that describe the match.
    MatchObj* mo_template;

    // Called after a delay in seconds; returns how long to wait before
    // calling again.  The parameter is "userdata".
    time_t (*timer_callback)(void*);

    // FIELDS THAT AFFECT THE RUNNING SOLVER ON CALLBACK
    // =================================================

    // Bail out ASAP.
    anbool quit_now;

    // SOLVER OUTPUTS
    // ==============
    // NOTE: these are only incremented, not initialized.  It's up to you to set
    // them to zero before calling, if you're starting from scratch.
    // See solver_reset_counters().
    int numtries;
    int nummatches;
    int numscaleok;
    // the last field object examined
    int last_examined_object;
    // number of quads skipped because of cxdx constraints.
    int num_cxdx_skipped;
    int num_meanx_skipped;
    // number of matches skipped due to RA,Dec bounds constraints.
    int num_radec_skipped;
    // 
    int num_abscale_skipped;
    // The number of times we ran verification on a quad.
    int num_verified;

    // INTERNAL PARAMETERS; DO NOT MODIFY
    // ==================================
    // The index we're currently dealing with.
    index_t* index;

    // The extreme limits of quad size, for all indexes, in pixels^2.
    double minminAB2;
    double maxmaxAB2;

    // The relative noise of the current index, squared:
    // square( index->index_jitter / index->index_scale_lower )
    double rel_index_noise2;

    // The relative noise of the current quad, squared:
    double rel_field_noise2;

    double abscale_low;
    double abscale_high;

    // Field limits, in pixels.
    double field_minx, field_maxx, field_miny, field_maxy;
    // Distance in pixels across the diagonal of the field
    double field_diag;

    // If the index has the property that cx <= dx, then how much of a margin do we
    // have to add before we can safely assume that a permutation of a quad's code
    // can't be in the index?
    double cxdx_margin;

    // How long has this been going on? (CPU time)
    double starttime;
    double timeused;

    // Best match so far
    double   best_logodds;
    MatchObj best_match;
    index_t* best_index;
    anbool     best_match_solves;
    anbool     have_best_match;

    // Cached data about this field, for verify_hit().
    verify_field_t* vf;
};
typedef struct solver_t solver_t;

solver_t* solver_new();

void solver_set_default_values(solver_t* solver);

/**
 Returns the assumed field positional uncertainty ("jitter")
 in pixels.
 */
double solver_get_field_jitter(const solver_t* solver);

/**
 Sets the log-odds ratio for "recording" a proposed solution.  NOTE,
 'recording' means calling your callback, where you can decide what to
 do with it; thus it's probably your "solve" threshold.

 Default: 0, which means your callback gets called for every match.

 Suggested value: log(1e9) or log(1e12).
 */
void solver_set_keep_logodds(solver_t* solver, double logodds);

// back-compate
#define solver_set_record_logodds solver_set_keep_logodds

/**
 Sets the "parity" or "flip" of the image.

 PARITY_NORMAL: det(CD) < 0
 PARITY_FLIP:   det(CD) > 0
 PARITY_BOTH (default): try both.

 Return 0 on success, -1 on invalid parity value.
 */
int solver_set_parity(solver_t* solver, int parity);

/**
 Returns the pixel position of the center of the field, defined
 to be the mean of the min and max position.
 (field_maxx + field_maxy)/2
 */
void solver_get_field_center(const solver_t* solver, double* px, double* py);

/**
 Returns the maximum field size expected, in arcsec.

 This is simply the maximum pixel scale * maximum radius (on the
 diagonal)
 */
double solver_get_max_radius_arcsec(const solver_t* solver);

/**
 Returns the best match found after a solver_run() call.

 This is just &(solver->best_match)
 */
MatchObj* solver_get_best_match(solver_t* solver);

/**
 Did the best match solve?

 Returns solver->best_match_solved.
 */
anbool solver_did_solve(const solver_t* solver);

/**
 Returns solver->best_index->indexname.

 (Should be equal to solver->best_match->index->indexname.
 */
const char* solver_get_best_match_index_name(const solver_t* solver);

/**
 Returns the lower/upper bounds of pixel scale that will be searched,
 in arcsec/pixel.
 */
double solver_get_pixscale_low(const solver_t* solver);
double solver_get_pixscale_high(const solver_t* solver);

/**
 Sets the range of quad sizes to try in the image.  This will be
 further tightened, if possible, given the range of quad sizes in the
 index and the pixel scale estimates.

 This avoids looking at tiny quads in the image, because if matches
 are generated they are difficult to verify (a tiny bit of noise in
 the quad position translates to a lot of positional noise in stars
 far away).

 Min and max are in pixels.

 Sets quadsize_min, quadsize_max fields.

 Recommended: ~ 10% of smaller image dimension; 100% of image diagonal.
 */
void solver_set_quad_size_range(solver_t* solver, double qmin, double qmax);

/**
 Same as solver_set_quad_size_range(), but specified in terms of
 fraction of the smaller image dimension (for lower) and the diagonal
 (for upper).

 Recommended: min=0.1, max=1
 */
void solver_set_quad_size_fraction(solver_t* solver, double qmin, double qmax);

/**
 Returns the range of quad sizes that could be matched, given the
 current settings of pixel scale and image quad size.

 Returns minimum pixel scale * minimum quad size and
 maximum pixel scale * maximum quad size.
 */
void solver_get_quad_size_range_arcsec(const solver_t* solver, double* qmin, double* qmax);

void solver_free(solver_t*);

/**
 Tells the solver which field of stars it's going to be solving.

 The solver_t* takes ownership of the *field*; it will be freed upon
 solver_free() or solver_cleanup_field() or a new solver_set_field().
 */
void solver_set_field(solver_t* s, starxy_t* field);

starxy_t* solver_get_field(solver_t* solver);

void solver_reset_field_size(solver_t* s);

/**
 Tells the solver to only accept matches within "radius_deg" (in
 degrees) of the given "ra","dec" point (also in degrees).

 This is, each star comprising the quad must be within that circle.
 */
void solver_set_radec(solver_t* s, double ra, double dec, double radius_deg);

void solver_clear_radec(solver_t* s);

/**
 Tells the solver the pixel coordinate range of the image to be
 solved.  If not set, this will be computed based on the bounds of the
 stars within the field (an underestimate).
 */
void solver_set_field_bounds(solver_t* s, double xlo, double xhi, double ylo, double yhi);

/**
 Reset everything associated with solving a particular field.

 (renamed from solver_new_field)
 */
void solver_cleanup_field(solver_t*);

/**
 get field w,h
 */
double solver_field_width(const solver_t* t);
double solver_field_height(const solver_t* t);


void solver_add_index(solver_t* solver, index_t* index);
void solver_clear_indexes(solver_t* solver);

int solver_n_indices(const solver_t* solver);
index_t* solver_get_index(const solver_t* solver, int i);

void solver_verify_sip_wcs(solver_t* solver, sip_t* sip); //, MatchObj* mo);

void solver_run(solver_t* solver);

#define SOLVER_TWEAK2_AVAILABLE 1
void solver_tweak2(solver_t* solver, MatchObj* mo, int order, sip_t* verifysip);

void solver_cleanup(solver_t* solver);

// Call this before solver_inject_match(), solver_verify_sip_wcs() or solver_run().
// (or it will get called automatically)
void solver_preprocess_field(solver_t* sp);
// Call this after solver_inject_match() or solver_run().
// (or it will get called when you solver_free())
void solver_free_field(solver_t* sp);

void solver_inject_match(solver_t* solver, MatchObj* mo, sip_t* sip);
void solver_compute_quad_range(const solver_t* solver, const index_t* index, double*, double*);

/**
 Resets the "numtries", "nummatches", etc counters, as well as
 "quitnow".
 */
void solver_reset_counters(solver_t* t);

/**
 Clears the "best_match_solves", "have_best_match", etc fields.
 */
void solver_reset_best_match(solver_t* sp);

void solver_print_to(const solver_t* sp, FILE* stream);

void solver_log_params(const solver_t* sp);

#endif
