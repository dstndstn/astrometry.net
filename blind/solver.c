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

#include <stdio.h>
#include <sys/param.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdarg.h>

#include "ioutils.h"
#include "mathutil.h"
#include "matchobj.h"
#include "solver.h"
#include "verify.h"
#include "tic.h"
#include "solvedclient.h"
#include "solvedfile.h"
#include "blind_wcs.h"
#include "keywords.h"
#include "log.h"
#include "pquad.h"
#include "kdtree.h"

#if TESTING
#define DEBUGSOLVER 1
#define TRY_ALL_CODES test_try_all_codes
void test_try_all_codes(pquad* pq,
                        int* fieldstars, int dimquad,
                        solver_t* solver, double tol2);
#else
#define TRY_ALL_CODES try_all_codes
#endif


static const int A = 0, B = 1, C = 2, D = 3;

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

double solver_field_width(solver_t* s) {
	return s->field_maxx - s->field_minx;
}
double solver_field_height(solver_t* s) {
	return s->field_maxy - s->field_miny;
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
    s->rel_index_noise2 = square(index->meta.index_jitter / index->meta.index_scale_lower);
}

void solver_set_field(solver_t* s, starxy_t* field) {
    s->fieldxy = field;
    // FIXME -- compute size of field, etc?
}

void solver_new_field(solver_t* solver) {
    solver_reset_best_match(solver);
    solver_free_field(solver);
    solver->fieldxy = NULL;
    solver->numtries = 0;
    solver->nummatches = 0;
    solver->numscaleok = 0;
    solver->num_cxdx_skipped = 0;
    solver->num_verified = 0;
    solver->last_examined_object = 0;
    solver->quit_now = FALSE;
}

void solver_verify_sip_wcs(solver_t* solver, sip_t* sip) {
    int i, nindexes;
    MatchObj mo;

	if (!solver->vf)
		solver_preprocess_field(solver);

    // fabricate a match and inject it into the solver.
    set_matchobj_template(solver, &mo);
    memcpy(&(mo.wcstan), &(sip->wcstan), sizeof(tan_t));
    mo.wcs_valid = TRUE;
    mo.scale = sip_pixel_scale(sip);
    set_center_and_radius(solver, &mo, NULL, sip);
    solver->distance_from_quad_bonus = FALSE;

    nindexes = pl_size(solver->indexes);
    for (i=0; i<nindexes; i++) {
        index_t* index = pl_get(solver->indexes, i);
        set_index(solver, index);
        solver_inject_match(solver, &mo, sip);
    }
}

void solver_add_index(solver_t* solver, index_t* index) {
    pl_append(solver->indexes, index);
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
	double scalefudge = 0.0; // in pixels

	if (sp->funits_upper != 0.0) {
		*minAB = index->meta.index_scale_lower / sp->funits_upper;

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
		scalefudge = index->meta.index_scale_upper * M_SQRT1_2 *
            sp->codetol / sp->funits_upper;
		*minAB -= scalefudge;
	}
	if (sp->funits_lower != 0.0) {
		*maxAB = index->meta.index_scale_upper / sp->funits_lower;
		*maxAB += scalefudge;
	}
}

static void try_all_codes(pquad* pq,
                          int* fieldstars, int dimquad,
                          solver_t* solver, double tol2);

static void try_all_codes_2(int* fieldstars, int dimquad,
                            double* code, solver_t* solver,
                            bool current_parity, double tol2);

static void try_permutations(int* origstars, int dimquad, double* origcode,
							 solver_t* solver, bool current_parity,
							 double tol2,
							 int* stars,
							 double* code,
							 int firststar,
							 int star,
							 bool* placed,
							 bool first,
							 kdtree_qres_t** presult);

static void resolve_matches(kdtree_qres_t* krez, double *query, double *field,
                            int* fstars, int dimquads,
                            solver_t* solver, bool current_parity);

static int solver_handle_hit(solver_t* sp, MatchObj* mo, sip_t* sip, bool fake_match);

static void check_scale(pquad* pq, solver_t* solver) {
	double dx, dy;
    
    dx = (starxy_getx(solver->fieldxy, pq->fieldB) -
          starxy_getx(solver->fieldxy, pq->fieldA));
    dy = (starxy_gety(solver->fieldxy, pq->fieldB) -
          starxy_gety(solver->fieldxy, pq->fieldA));

	pq->scale = dx * dx + dy * dy;
	if ((pq->scale < solver->minminAB2) ||
			(pq->scale > solver->maxmaxAB2)) {
		pq->scale_ok = FALSE;
		return;
	}
	pq->costheta = (dy + dx) / pq->scale;
	pq->sintheta = (dy - dx) / pq->scale;
	pq->rel_field_noise2 = (solver->verify_pix * solver->verify_pix) / pq->scale;
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


static void find_field_boundaries(solver_t* solver) {
	if ((solver->field_minx == solver->field_maxx) 
			|| (solver->field_miny == solver->field_maxy)) {
		int i;
		for (i = 0; i < starxy_n(solver->fieldxy); i++) {
			solver->field_minx = MIN(solver->field_minx, field_getx(solver, i));
			solver->field_maxx = MAX(solver->field_maxx, field_getx(solver, i));
			solver->field_miny = MIN(solver->field_miny, field_gety(solver, i));
			solver->field_maxy = MAX(solver->field_maxy, field_gety(solver, i));
		}
	}
	solver->field_diag = hypot(solver->field_maxy - solver->field_miny,
	                       solver->field_maxx - solver->field_minx);
}

void solver_preprocess_field(solver_t* solver) {
	find_field_boundaries(solver);
	// precompute a kdtree over the field
	solver->vf = verify_field_preprocess(solver->fieldxy);
}

void solver_free_field(solver_t* solver) {
	if (solver->vf)
		verify_field_free(solver->vf);
	solver->vf = NULL;
}

void solver_resolve_correspondences(const solver_t* sp, MatchObj* mo) {
	int j;
    mo->corr_field_xy = dl_new(16);
    mo->corr_index_rd = dl_new(16);
    for (j=0; j<il_size(mo->corr_field); j++) {
        double ixyz[3];
        double iradec[2];
        int iindex, ifield;

        ifield = il_get(mo->corr_field, j);
        iindex = il_get(mo->corr_index, j);
        assert(ifield >= 0);
        assert(ifield < starxy_n(sp->fieldxy));

        dl_append(mo->corr_field_xy, starxy_getx(sp->fieldxy, ifield));
        dl_append(mo->corr_field_xy, starxy_gety(sp->fieldxy, ifield));

        startree_get(sp->index->starkd, iindex, ixyz);
        xyzarr2radecdegarr(ixyz, iradec);
        dl_append(mo->corr_index_rd, iradec[0]);
        dl_append(mo->corr_index_rd, iradec[1]);
    }
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
 A somewhat tricky recursive function: stars A and B have already been chosen,
 so the code coordinate system has been fixed, and we've already determined
 which other stars will create valid codes (ie, are in the "box").  Now we
 want to build features using all sets of valid stars (without permutations).

 pq - data associated with the AB pair.
 field - the array of field star numbers
 fieldoffset - offset into the field array where we should add the first star
 n_to_add - number of stars to add
 adding - the star we're currently adding; in [0, n_to_add).
 fieldtop - the maximum field star number to build quads out of.
 dimquad, solver, tol2 - passed to try_all_codes.
 */
static void add_stars(pquad* pq, int* field, int fieldoffset,
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
	int i;
	double usertime, systime;
	time_t next_timer_callback_time = time(NULL) + 1;
	pquad* pquads;
	int num_indexes;
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

	num_indexes = pl_size(solver->indexes);
	{
		double minAB2s[num_indexes];
		double maxAB2s[num_indexes];
		solver->minminAB2 = HUGE_VAL;
		solver->maxmaxAB2 = -HUGE_VAL;
		for (i = 0; i < num_indexes; i++) {
			double minAB=0, maxAB=0;
			index_t* index = pl_get(solver->indexes, i);
			// The limits on the size of quads that we try to match, in pixels.
			// Derived from index_scale_* and funits_*.
			solver_compute_quad_range(solver, index, &minAB, &maxAB);
			minAB2s[i] = square(minAB);
			maxAB2s[i] = square(maxAB);
			solver->minminAB2 = MIN(solver->minminAB2, minAB2s[i]);
			solver->maxmaxAB2 = MAX(solver->maxmaxAB2, maxAB2s[i]);

			if (index->meta.cx_less_than_dx) {
				solver->cxdx_margin = 1.5 * solver->codetol;
				// FIXME die horribly if the indexes have differing cx_less_than_dx
			}
		}
		solver->minminAB2 = MAX(solver->minminAB2, square(solver->quadsize_min));
        if (solver->quadsize_max != 0.0)
            solver->maxmaxAB2 = MIN(solver->maxmaxAB2, square(solver->quadsize_max));
		logverb("Quad scale range: [%g, %g] pixels\n", sqrt(solver->minminAB2), sqrt(solver->maxmaxAB2));

		pquads = calloc(numxy * numxy, sizeof(pquad));

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
					pq->inbox = malloc(numxy * sizeof(bool));
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

			debug("Trying newpoint=%i\n", newpoint);

			// Give our caller a chance to cancel us midway. The callback
			// returns how long to wait before calling again.

			if (solver->timer_callback) {
				time_t delay;
                time_t now = time(NULL);
				if (now > next_timer_callback_time) {
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
				pq->inbox = malloc(numxy * sizeof(bool));
				pq->xy = malloc(numxy * 2 * sizeof(double));
				// -try all stars up to "newpoint"...
				assert(sizeof(bool) == 1);
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
                dimquads = quadfile_dimquads(index->quads);

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

                    solver->rel_field_noise2 = pq->rel_field_noise2;

					for (i = 0; i < pl_size(solver->indexes); i++) {
                        int dimquads;
						index_t* index = pl_get(solver->indexes, i);
						if ((pq->scale < minAB2s[i]) ||
					        (pq->scale > maxAB2s[i]))
							continue;
                        set_index(solver, index);
						dimquads = quadfile_dimquads(index->quads);

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

static void try_all_codes(pquad* pq,
                          int* fieldstars, int dimquad,
                          solver_t* solver, double tol2) {
    int dimcode = (dimquad - 2) * 2;
    double code[dimcode];
    double swapcode[dimcode];
    int i;

    solver->numtries++;

    debug("  trying quad [");
	for (i=0; i<dimquad; i++) {
		debug("%s%i", (i?" ":""), fieldstars[i]);
	}
	debug("]\n");

    for (i=0; i<dimcode/2; i++) {
        code[2*i  ] = getx(pq->xy, fieldstars[C+i]);
        code[2*i+1] = gety(pq->xy, fieldstars[C+i]);
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
        int i;
        // swap CX <-> CY, DX <-> DY.
        for (i=0; i<dimcode/2; i++) {
            swapcode[2*i+0] = code[2*i+1];
            swapcode[2*i+1] = code[2*i+0];
        }

		debug("    trying reverse parity: code=[");
		for (i=0; i<dimcode; i++)
			debug("%s%g", (i?", ":""), swapcode[i]);
		debug("].\n");

		try_all_codes_2(fieldstars, dimquad, swapcode, solver, TRUE, tol2);
	}
}

static void try_all_codes_2(int* fieldstars, int dimquad,
                            double* code, solver_t* solver,
                            bool current_parity, double tol2) {
	int i;
	kdtree_qres_t* result = NULL;
	int flipab;
    int dimcode = (dimquad - 2) * 2;

	for (flipab=0; flipab<2; flipab++) {
		double flipcode[dimcode];
		double* origcode;
		double searchcode[dimcode];
		int stars[dimquad];
		bool placed[dimquad];

		for (i=0; i<dimquad; i++)
			placed[i] = FALSE;
		
		if (!flipab) {
			origcode = code;
			stars[0] = fieldstars[0];
			stars[1] = fieldstars[1];
		} else {
			for (i=0; i<dimcode; i++)
				flipcode[i] = 1.0 - code[i];
			origcode = flipcode;

			stars[0] = fieldstars[1];
			stars[1] = fieldstars[0];
		}

		try_permutations(fieldstars, dimquad, origcode, solver, current_parity,
						 tol2, stars, searchcode, 2, 2, placed, TRUE, &result);
		if (unlikely(solver->quit_now))
			break;
	}

	kdtree_free_query(result);
}


static void try_permutations(int* origstars, int dimquad, double* origcode,
							 solver_t* solver, bool current_parity,
							 double tol2,
							 int* stars,
							 double* code,
							 int firststar,
							 int star,
							 bool* placed,
							 bool first,
							 kdtree_qres_t** presult) {
	int i;
	int options = KD_OPTIONS_SMALL_RADIUS | KD_OPTIONS_COMPUTE_DISTS |
		KD_OPTIONS_NO_RESIZE_RESULTS;
	// We have, say, three stars to be placed, C,D,E in position "star".
	// Stars can't be reused so we check in the "placed" array to see if it has
	// already been placed.
	// If we're not placing the first star, we ensure that the cx<=dx criterion
	// is satisfied (if the index has that property).

	// Look for "unplaced" stars.
	for (i=firststar; i<dimquad; i++) {
		if (placed[i])
			continue;
		if (!first && solver->index->meta.cx_less_than_dx &&
			(code[2 * (star - 1 - 2)] > origcode[2 * (i - 2)] + solver->cxdx_margin))
			continue;
		//solver->num_cxdx_skipped++;

		stars[star] = origstars[i];
		code[2*(star-2)+0] = origcode[2*(i-2)+0];
		code[2*(star-2)+1] = origcode[2*(i-2)+1];

		if (star == dimquad-1) {
			double pixvals[dimquad*2];
			int j;

			for (j=0; j<dimquad; j++) {
                setx(pixvals, j, field_getx(solver, stars[j]));
                sety(pixvals, j, field_gety(solver, stars[j]));
            }

            options |= KD_OPTIONS_USE_SPLIT;

			*presult = kdtree_rangesearch_options_reuse(solver->index->codekd->tree,
														*presult, code, tol2, options);

			//debug("      trying ABCD = [%i %i %i %i]: %i results.\n", fstars[A], fstars[B], fstars[C], fstars[D], result->nres);

			if ((*presult)->nres)
				resolve_matches(*presult, code, pixvals, stars, dimquad, solver, current_parity);

			if (unlikely(solver->quit_now))
				return;

		} else {
			placed[i] = TRUE;

			try_permutations(origstars, dimquad, origcode, solver,
							 current_parity, tol2, stars, code, firststar,
							 star+1, placed, FALSE, presult);

			placed[i] = FALSE;
		}
	}
}

// "field" contains the xy pixel coordinates of stars A,B,C,D.
static void resolve_matches(kdtree_qres_t* krez, double *query, double *field,
                            int* fieldstars, int dimquads,
                            solver_t* solver, bool current_parity) {
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

		solver->nummatches++;
		thisquadno = krez->inds[jj];
		quadfile_get_stars(solver->index->quads, thisquadno, star);
        for (i=0; i<dimquads; i++)
            startree_get(solver->index->starkd, star[i], starxyz + 3*i);

		debug("        stars [");
		for (i=0; i<dimquads; i++)
			debug("%s%i", (i?" ":""), star[i]);
		debug("]\n;");

		// FIXME -- could compute position here and compare with
		//  --ra,dec,radius !!

		// FIXME -- could compute approximate scale here (based on AB
		// distance), before computing full WCS solution

		// compute TAN projection from the matching quad alone.
		if (blind_wcs_compute(starxyz, field, dimquads, &wcs, &scale)) {
            // bad quad.
            continue;
        }
		arcsecperpix = scale * 3600.0;

		// FIXME - should there be scale fudge here?
		if (arcsecperpix > solver->funits_upper ||
		        arcsecperpix < solver->funits_lower) {
			debug("          bad scale.\n");
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

		memcpy(mo.quadpix, field, 2 * dimquads * sizeof(double));
		memcpy(mo.quadxyz, starxyz, 3 * dimquads * sizeof(double));

		set_center_and_radius(solver, &mo, &(mo.wcstan), NULL);

		if (solver_handle_hit(solver, &mo, NULL, FALSE))
			solver->quit_now = TRUE;

        if (mo.corr_field)
            il_free(mo.corr_field);
        if (mo.corr_index)
            il_free(mo.corr_index);

		if (unlikely(solver->quit_now))
			return;
	}
}

void solver_inject_match(solver_t* solver, MatchObj* mo, sip_t* sip) {
	solver_handle_hit(solver, mo, sip, TRUE);
}

static int solver_handle_hit(solver_t* sp, MatchObj* mo, sip_t* sip, bool fake_match) {
	double match_distance_in_pixels2;
    bool solved;

	mo->indexid = sp->index->meta.indexid;
	mo->healpix = sp->index->meta.healpix;
	mo->hpnside = sp->index->meta.hpnside;
	mo->wcstan.imagew = sp->field_maxx;
	mo->wcstan.imageh = sp->field_maxy;

	match_distance_in_pixels2 = square(sp->verify_pix) +
		square(sp->index->meta.index_jitter / mo->scale);

	mo->dimquads = quadfile_dimquads(sp->index->quads);

	verify_hit(sp->index->starkd, sp->index->meta.cutnside,
			   mo, sip, sp->vf, match_distance_in_pixels2,
	           sp->distractor_ratio, sp->field_maxx, sp->field_maxy,
	           sp->logratio_bail_threshold, sp->logratio_record_threshold,
			   sp->logratio_stoplooking,
			   sp->distance_from_quad_bonus, fake_match);

	mo->nverified = sp->num_verified++;

	if (mo->logodds >= sp->best_logodds)
		sp->best_logodds = mo->logodds;

	if (!sp->have_best_match || (mo->logodds > sp->best_match.logodds)) {
		logverb("Got a new best match: logodds %g.\n", mo->logodds);
		// FIXME -- set logodds_toaccept to this so that the mo is valid here?
		memcpy(&(sp->best_match), mo, sizeof(MatchObj));
		sp->have_best_match = TRUE;
		sp->best_index = sp->index;
	}

	if (mo->logodds < sp->logratio_record_threshold)
		return FALSE;

	update_timeused(sp);
	mo->timeused = sp->timeused;

	if (sp->set_crpix) {
		double crpix[2];
		tan_t wcs2;
		tan_t wcs3;
		if (sp->set_crpix_center) {
			crpix[0] = 1 + 0.5 * solver_field_width(sp);
			crpix[1] = 1 + 0.5 * solver_field_height(sp);
		} else {
			crpix[0] = sp->crpix[0];
			crpix[1] = sp->crpix[1];
		}
		blind_wcs_move_tangent_point(mo->quadxyz, mo->quadpix, mo->dimquads, crpix, &(mo->wcstan), &wcs2);
		blind_wcs_move_tangent_point(mo->quadxyz, mo->quadpix, mo->dimquads, crpix, &wcs2, &wcs3);
		memcpy(&(mo->wcstan), &wcs3, sizeof(tan_t));
		/*
		 Good test case:
		 http://antwrp.gsfc.nasa.gov/apod/image/0912/Geminid2007_pacholka850wp.jpg
		 solve-field --backend-config backend.cfg Geminid2007_pacholka850wp.xy \
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
	solver->funits_upper = HUGE_VAL;
	solver->logratio_bail_threshold = log(1e-100);
	solver->logratio_stoplooking = HUGE_VAL;
	solver->parity = DEFAULT_PARITY;
	solver->codetol = DEFAULT_CODE_TOL;
    solver->distractor_ratio = DEFAULT_DISTRACTOR_RATIO;
    solver->verify_pix = DEFAULT_VERIFY_PIX;
}

void solver_clear_indexes(solver_t* solver) {
	pl_remove_all(solver->indexes);
    solver->index = NULL;
}

void solver_cleanup(solver_t* solver) {
	solver_free_field(solver);
	pl_free(solver->indexes);
    solver->indexes = NULL;
}

void solver_free(solver_t* solver) {
    if (!solver) return;
	solver_cleanup(solver);
    free(solver);
}
