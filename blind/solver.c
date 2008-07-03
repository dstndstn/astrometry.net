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
#include <sys/mman.h>
#include <sys/param.h>
#include <errno.h>
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
#include "svd.h"
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
	*x = sp->field[2*index];
	*y = sp->field[2*index+1];
}

static double field_getx(solver_t* sp, int index) {
	return sp->field[2*index];
}
static double field_gety(solver_t* sp, int index) {
	return sp->field[2*index+1];
}

static void update_timeused(solver_t* sp) {
	double usertime, systime;
	get_resource_stats(&usertime, &systime, NULL);
	sp->timeused = (usertime + systime) - sp->starttime;
	if (sp->timeused < 0.0)
		sp->timeused = 0.0;
}

void solver_reset_best_match(solver_t* sp) {
    // we don't really care about very bad best matches...
	sp->best_logodds = 0;
	memset(&(sp->best_match), 0, sizeof(MatchObj));
	sp->best_index = NULL;
	sp->best_index_num = 0;
	sp->best_match_solves = FALSE;
	sp->have_best_match = FALSE;
}

/*
  Reorder the field stars by placing a grid over the field, sorting by
  brightness within each grid cell, then making sweeps over the grid, taking
  the Nth brightest star in each cell in the Nth sweep.  (And sort the stars
  within a sweep by their absolute brightness.)
 */
void solver_uniformize_field(solver_t* solver, int NX, int NY) {
	il** lists;
	int i, ix, iy;
	double xstep, ystep;
	double* newfield;
	int Ntotal;

	// compute {min,max}{x,y}
	find_field_boundaries(solver);

	assert(NX > 0);
	assert(NY > 0);
	lists = malloc(NX * NY * sizeof(il*));
	for (i=0; i<(NX*NY); i++)
		lists[i] = il_new(16);
	newfield = calloc(solver->nfield * 2, sizeof(double));

	// grid cell size
	xstep = (solver->field_maxx - solver->field_minx) / (double)NX;
	ystep = (solver->field_maxy - solver->field_miny) / (double)NY;
	assert(xstep > 0.0);
	assert(ystep > 0.0);

	// place stars in grid cells.
	for (i=0; i<solver->nfield; i++) {
		ix = floor((field_getx(solver, i) - solver->field_minx) / xstep);
		iy = floor((field_gety(solver, i) - solver->field_miny) / ystep);
		assert(ix >= 0);
		assert(iy >= 0);
		assert(ix < NX);
		assert(iy < NY);
		il_append(lists[iy * NX + ix], i);
	}
	// since the stars were originally sorted by brightness, each grid
	// cell is also sorted by brightness.

	// reverse each list so that we can pop() bright stars off the back.
	for (i=0; i<(NX*NY); i++)
		il_reverse(lists[i]);

	// sweep through the cells...
	Ntotal = 0;
	for (;;) {
		bool allempty = TRUE;
		il* thissweep = il_new(16);

		for (i=0; i<(NX*NY); i++) {
			if (!il_size(lists[i]))
				continue;
			allempty = FALSE;
			il_insert_ascending(thissweep, il_pop(lists[i]));
		}

		for (i=0; i<il_size(thissweep); i++) {
			int ind = il_get(thissweep, i);
			setx(newfield, Ntotal, field_getx(solver, ind));
			sety(newfield, Ntotal, field_gety(solver, ind));
			Ntotal++;
		}

		il_free(thissweep);
		if (allempty)
			break;
	}

	assert(Ntotal == solver->nfield);

	for (i=0; i<(NX*NY); i++)
		il_free(lists[i]);
	free(lists);

	memcpy(solver->field, newfield, solver->nfield * 2 * sizeof(double));
	free(newfield);
}

void solver_transform_corners(solver_t* solver, MatchObj* mo) {
	// transform the corners of the field...
	tan_pixelxy2xyzarr(&(mo->wcstan), solver->field_minx, solver->field_miny, mo->sMin);
	tan_pixelxy2xyzarr(&(mo->wcstan), solver->field_maxx, solver->field_maxy, mo->sMax);
	tan_pixelxy2xyzarr(&(mo->wcstan), solver->field_minx, solver->field_maxy, mo->sMinMax);
	tan_pixelxy2xyzarr(&(mo->wcstan), solver->field_maxx, solver->field_miny, mo->sMaxMin);
	// center and radius...
	star_midpoint(mo->center, mo->sMin, mo->sMax);
	mo->radius = sqrt(distsq(mo->center, mo->sMin, 3));

    assert(isfinite(mo->radius));
    assert(isfinite(mo->sMin[0]));
    assert(isfinite(mo->sMax[0]));
    assert(isfinite(mo->sMinMax[0]));
    assert(isfinite(mo->sMaxMin[0]));
    assert(isfinite(mo->center[0]));
}

void solver_compute_quad_range(solver_t* sp, index_t* index,
		double* minAB, double* maxAB)
{
	double scalefudge = 0.0; // in pixels

	if (sp->funits_upper != 0.0) {
		*minAB = index->index_scale_lower / sp->funits_upper;

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
		scalefudge = index->index_scale_upper * M_SQRT1_2 *
		             sp->codetol / sp->funits_upper;
		*minAB -= scalefudge;
	}
	if (sp->funits_lower != 0.0) {
		*maxAB = index->index_scale_upper / sp->funits_lower;
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
	double Ax, Ay, Bx, By, dx, dy;

	field_getxy(solver, pq->fieldA, &Ax, &Ay);
	field_getxy(solver, pq->fieldB, &Bx, &By);
	dx = Bx - Ax;
	dy = By - Ay;
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
		for (i = 0; i < solver->nfield; i++) {
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
	// precompute a kdtree over the field
	solver->vf = verify_field_preprocess(solver->field, solver->nfield);
	find_field_boundaries(solver);
}

void solver_free_field(solver_t* solver) {
	verify_field_free(solver->vf);
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

	memset(field, 0, sizeof(field));

	get_resource_stats(&usertime, &systime, NULL);
	solver->starttime = usertime + systime;

	numxy = solver->nfield;
	if (solver->endobj && (numxy > solver->endobj))
		numxy = solver->endobj;
	if (solver->startobj >= numxy)
		return;

	num_indexes = bl_size(solver->indexes);
	{
		double minAB2s[num_indexes];
		double maxAB2s[num_indexes];
		solver->minminAB2 = HUGE_VAL;
		solver->maxmaxAB2 = -HUGE_VAL;
		for (i = 0; i < num_indexes; i++) {
			double minAB, maxAB;
			index_t* index = pl_get(solver->indexes, i);
			// The limits on the size of quads that we try to match, in pixels.
			// Derived from index_scale_* and funits_*.
			solver_compute_quad_range(solver, index, &minAB, &maxAB);
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

                solver->index_num = i;
                solver->index = index;
                solver->rel_index_noise2 = square(index->index_jitter / index->index_scale_lower);
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
						solver->index_num = i;
						solver->index = index;
						solver->rel_index_noise2 = square(index->index_jitter / index->index_scale_lower);
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

static inline void set_xy(double* dest, int destind, double* src, int srcind) {
	setx(dest, destind, getx(src, srcind));
	sety(dest, destind, gety(src, srcind));
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

	//assert(solver->index->cx_less_than_dx

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
		if (!first && solver->index->cx_less_than_dx &&
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

	for (jj = 0; jj < krez->nres; jj++) {
		double starxyz[dimquads*3];
		double scale;
		double arcsecperpix;
		tan_t wcs;
        int i;

		solver->nummatches++;

		thisquadno = (int)krez->inds[jj];

		quadfile_get_stars(solver->index->quads, thisquadno, star);
        for (i=0; i<dimquads; i++)
            startree_get(solver->index->starkd, star[i], starxyz + 3*i);

		debug("        stars [");
		for (i=0; i<dimquads; i++)
			debug("%s%i", (i?" ":""), star[i]);
		debug("]\n;");

		// compute TAN projection from the matching quad alone.
		blind_wcs_compute(starxyz, field, dimquads, &wcs, &scale);
		arcsecperpix = scale * 3600.0;

		// FIXME - should there be scale fudge here?
		if (arcsecperpix > solver->funits_upper ||
		        arcsecperpix < solver->funits_lower) {
			debug("          bad scale.\n");
			continue;
		}

		solver->numscaleok++;

		if (solver->mo_template)
			memcpy(&mo, solver->mo_template, sizeof(MatchObj));
        else
            memset(&mo, 0, sizeof(MatchObj));

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
        for (i=0; i<dimquads; i++)
            mo.star[i] = star[i];
        for (i=0; i<dimquads; i++)
            mo.field[i] = fieldstars[i];

		memcpy(mo.quadpix, field, 2 * dimquads * sizeof(double));
		memcpy(mo.quadxyz, starxyz, 3 * dimquads * sizeof(double));

		if (index_has_ids(solver->index)) {
            for (i=0; i<dimquads; i++)
                mo.ids[i] = startree_get_starid(solver->index->starkd, star[i]);
		} else {
            for (i=0; i<dimquads; i++)
                mo.ids[i] = 0;
		}

		solver_transform_corners(solver, &mo);

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
	int dimquads;

	mo->indexid = sp->index->indexid;
	mo->healpix = sp->index->healpix;
	mo->hpnside = sp->index->hpnside;
	mo->wcstan.imagew = sp->field_maxx;
	mo->wcstan.imageh = sp->field_maxy;

	match_distance_in_pixels2 = square(sp->verify_pix) +
		square(sp->index->index_jitter / mo->scale);

	dimquads = quadfile_dimquads(sp->index->quads);

	verify_hit(sp->index->starkd, mo, sip, sp->vf, match_distance_in_pixels2,
	           sp->distractor_ratio, sp->field_maxx, sp->field_maxy,
	           sp->logratio_bail_threshold, sp->distance_from_quad_bonus,
			   dimquads, fake_match);
	mo->nverified = sp->num_verified++;

	if (mo->logodds >= sp->best_logodds)
		sp->best_logodds = mo->logodds;

	if (!sp->have_best_match || (mo->logodds > sp->best_match.logodds)) {
		logverb("Got a new best match: logodds %g.\n", mo->logodds);

        //free_best_match_extras(sp);

		//print_match(bp, mo);
		memcpy(&(sp->best_match), mo, sizeof(MatchObj));
		sp->have_best_match = TRUE;
		sp->best_index = sp->index;
		sp->best_index_num = sp->index_num;
	}

	if (mo->logodds < sp->logratio_record_threshold)
		return FALSE;

	update_timeused(sp);
	mo->timeused = sp->timeused;

	solved = sp->record_match_callback(mo, sp->userdata);

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
	solver->parity = DEFAULT_PARITY;
	solver->codetol = DEFAULT_CODE_TOL;
}

void solver_clear_indexes(solver_t* solver) {
	pl_remove_all(solver->indexes);
}

void solver_cleanup(solver_t* solver) {
	pl_free(solver->indexes);
    solver->indexes = NULL;
}

void solver_free(solver_t* solver) {
    if (!solver) return;
	solver_cleanup(solver);
    free(solver);
}
