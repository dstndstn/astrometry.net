/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef SOLVER_H
#define SOLVER_H

#include <time.h>

#include "starutil.h"
#include "starxy.h"
#include "kdtree.h"
#include "bl.h"
#include "matchobj.h"
#include "quadfile.h"
#include "starkd.h"
#include "codekd.h"
#include "index.h"
#include "verify.h"
#include "sip.h"

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

	// Limits on the image pixel scale in [arcsec per pixel].
	double funits_lower;
	double funits_upper;

	// Callback; called for each match found whose log-odds ratio is above
	// "logratio_record_threshold".  The second parameter is "userdata".
	bool (*record_match_callback)(MatchObj*, void*);
	double logratio_record_threshold;

	// User data passed to the callbacks
	void* userdata;

	// Assume that stars far from the matched quad will have larger positional
	// variance?
	bool distance_from_quad_bonus;

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
	bool use_radec;
	double centerxyz[3];
	double r2;
	
	// During verification, if the log-odds ratio drops to this level, we bail out and
	// assume it's not a match.  Default log(1e-100).
	double logratio_bail_threshold;

	// During verification, if the log-odds ratio rises above this level, we accept the
	// match and bail out.  Default: HUGE_VAL (ie, don't bail out: keep going to find the
	// maximum Bayes factor value).
	double logratio_stoplooking;

	// Number of field quads to try or zero for no limit.
	int maxquads;
	// Number of quad matches to try or zero for no limit.
	int maxmatches;

	// Force CRPIX to be the given point "crpix", or the center of the image?
	bool set_crpix;
	bool set_crpix_center;
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
	bool quit_now;

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
	bool     best_match_solves;
	bool     have_best_match;

	// Cached data about this field, for verify_hit().
	verify_field_t* vf;
};
typedef struct solver_t solver_t;

solver_t* solver_new();

void solver_set_default_values(solver_t* solver);

void solver_free(solver_t*);

/**
 Tells the solver which field of stars it's going to be solving.
 */
void solver_set_field(solver_t* s, starxy_t* field);

/**
 Tells the solver to only accept matches within "radius_deg" (in
 degrees) of the given "ra","dec" point (also in degrees).

 This is, each star comprising the quad must be within than circle.
 */
void solver_set_radec(solver_t* s, double ra, double dec, double radius_deg);

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

double solver_field_width(solver_t* t);
double solver_field_height(solver_t* t);


void solver_add_index(solver_t* solver, index_t* index);
void solver_clear_indexes(solver_t* solver);

void solver_verify_sip_wcs(solver_t* solver, sip_t* sip);
void solver_run(solver_t* solver);

void solver_cleanup(solver_t* solver);

// Call this before solver_inject_match(), solver_verify_sip_wcs() or solver_run().
// (or it will get called automatically)
void solver_preprocess_field(solver_t* sp);
// Call this after solver_inject_match() or solver_run().
// (or it will get called when you solver_free())
void solver_free_field(solver_t* sp);

void solver_inject_match(solver_t* solver, MatchObj* mo, sip_t* sip);
void solver_compute_quad_range(const solver_t* solver, const index_t* index, double*, double*);

// During solving, the solver records the indices of image-to-index
// correspondences.  This function uses those indices to find the actual
// pixel and RA,Dec positions of the corresponding stars.
void solver_resolve_correspondences(const solver_t* solver, MatchObj* mo);

/**
 Resets the "numtries", "nummatches", etc counters, as well as
 "quitnow".
 */
void solver_reset_counters(solver_t* t);

/**
 Clears the "best_match_solves", "have_best_match", etc fields.
 */
void solver_reset_best_match(solver_t* sp);

#endif
