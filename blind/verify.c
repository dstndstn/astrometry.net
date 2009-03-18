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

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/param.h>

#include "verify.h"
#include "permutedsort.h"
#include "mathutil.h"
#include "keywords.h"
#include "log.h"

#define DEBUGVERIFY 1
#if DEBUGVERIFY
//#define debug(args...) fprintf(stderr, args)
#else
#define debug(args...)
#endif

/*
 This gets called once for each field before verification begins.
 We build a kdtree out of the field stars (in pixel space) which will be
 used during verification to find nearest-neighbours.
 */
verify_field_t* verify_field_preprocess(const starxy_t* fieldxy) {
    verify_field_t* vf;
    int Nleaf = 5;

    vf = malloc(sizeof(verify_field_t));
    if (!vf) {
        fprintf(stderr, "Failed to allocate space for a verify_field_t().\n");
        return NULL;
    }

    vf->field = fieldxy;

    // Note: kdtree type: I tried U32 (duu) but it was marginally slower.
    // I didn't try U16 (dss) because we need a fair bit of accuracy here.

    // Make a copy of the field objects, because we're going to build a
    // kdtree out of them and that shuffles their order.
    vf->fieldcopy = starxy_copy_xy(fieldxy);
    if (!vf->fieldcopy) {
        fprintf(stderr, "Failed to copy the field.\n");
        free(vf);
        return NULL;
    }

    // Build a tree out of the field objects (in pixel space)
    vf->ftree = kdtree_build(NULL, vf->fieldcopy, starxy_n(vf->field),
                             2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);

    return vf;
}

/*
 This gets called after all verification calls for a field are finished;
 we cleanup the data structures we created in the verify_field_preprocess()
 function.
 */
void verify_field_free(verify_field_t* vf) {
    if (!vf)
        return;
    kdtree_free(vf->ftree);
    free(vf->fieldcopy);
    free(vf);
    return;
}

static int get_index_stars(const double* fieldcenter, double fieldr2,
                           const startree_t* skdt, const sip_t* sip, const tan_t* tan,
                           double fieldW, double fieldH,
                           double** p_indexpix, int** p_starids, int* p_nindex) {
    int options = 0;
	kdtree_qres_t* res;
	double* indexpix;
    int i, NI;
    int* sweep;
    int* perm;
    int* starid;

	assert(skdt->sweep);

    // kdtree search options
    options |= KD_OPTIONS_SMALL_RADIUS;
    options |= KD_OPTIONS_USE_SPLIT;

	// find all the index stars that are inside the circle that bounds the field.
	// 1.01 is a little safety factor.
	res = kdtree_rangesearch_options(skdt->tree, fieldcenter, fieldr2 * 1.01, options);
	assert(res);
    debug("Found %i index stars.\n", res->nres);
    
	// Project index stars into pixel space.
	indexpix = malloc(res->nres * 2 * sizeof(double));
	NI = 0;
	for (i=0; i<res->nres; i++) {
		double x, y;
        if (sip) {
            if (!sip_xyzarr2pixelxy(sip, res->results.d + i*3, &x, &y))
                continue;
        } else {
            if (!tan_xyzarr2pixelxy(tan, res->results.d + i*3, &x, &y))
                continue;
        }
		if ((x < 0) || (y < 0) || (x >= fieldW) || (y >= fieldH))
			continue;

		// Here we compact the "res" arrays so that when we're done,
		// the NI indices that are inside the field are in the first
		// NI elements of the res->{results,inds} arrays.
		res->inds[NI] = res->inds[i];
		indexpix[NI*2  ] = x;
		indexpix[NI*2+1] = y;
		NI++;
	}
	indexpix = realloc(indexpix, NI * 2 * sizeof(double));
    debug("Found %i index stars in the field.\n", NI);

    if (!NI) {
		kdtree_free_query(res);
		free(indexpix);
        return -1;
    }

    // Each index star has a "sweep number" assigned during index building;
    // it roughly represents a local brightness ordering.
	sweep = malloc(NI * sizeof(int));
	for (i=0; i<NI; i++)
		sweep[i] = skdt->sweep[res->inds[i]];

    perm = permuted_sort(sweep, sizeof(int), compare_ints_asc, NULL, NI);

    starid = malloc(NI * sizeof(int));

    permutation_apply(perm, NI, indexpix, indexpix, 2 * sizeof(double));
    permutation_apply(perm, NI, res->inds, starid, sizeof(int));

    kdtree_free_query(res);
    free(sweep);
    free(perm);

    *p_indexpix = indexpix;
    *p_starids = starid;
    *p_nindex = NI;

    return 0;
}

//static void trim_index_stars()

/**
 If field objects are within "sigma" of each other (where sigma depends on the
 distance from the matched quad), then they are not very useful for verification.
 We filter out field stars within sigma of each other, taking only the brightest.

 Returns an array indicating which field stars should be kept.
 */
static bool* deduplicate_field_stars(verify_field_t* vf, double* sigma2s) {
    bool* keepers = NULL;
    int i, j, N;
    kdtree_qres_t* res;

    N = starxy_n(vf->field);
    keepers = malloc(N * sizeof(bool));
    for (i=0; i<N; i++) {
        double sxy[2];
        // free parameter!
        double nsigma2 = 1.0;
        keepers[i] = TRUE;
        starxy_get(vf->field, i, sxy);
        res = kdtree_rangesearch_nosort(vf->ftree, sxy, nsigma2 * sigma2s[i]);
        for (j=0; j<res->nres; j++) {
            if (res->inds[j] < i) {
                keepers[i] = FALSE;
                // DEBUG
                /*
                 double otherxy[2];
                 starxy_get(vf->field, res->inds[j], otherxy);
                 logverb("Field star %i at %g,%g: is close to field star %i at %g,%g.  dist is %g, sigma is %g\n",
                 i, sxy[0], sxy[1], res->inds[j], otherxy[0], otherxy[1],
                 sqrt(distsq(sxy, otherxy, 2)), sqrt(nsigma2 * sigma2s[i]));
                 */
                break;
            }
        }
        kdtree_free_query(res);
    }

    return keepers;
}


void verify_hit(startree_t* skdt, MatchObj* mo, sip_t* sip, verify_field_t* vf,
                double verify_pix2, double distractors,
                double fieldW, double fieldH,
                double logratio_tobail,
                bool do_gamma, int dimquads, bool fake_match) {
	int i;
	double* fieldcenter;
	double fieldr2;
	// number of stars in the index that are within the bounds of the field.
	int NI;
	double* indexpix;
    int* starids;

    int NF;

	double* bestprob = NULL;
	// quad center and radius
	double qc[2];
	double rquad2 = 0.0;
	double logprob_distractor;
	double logprob_background;
	double logodds = 0.0;
	int nmatch, nnomatch, nconflict;
	double bestlogodds;
	int bestnmatch, bestnnomatch, bestnconflict;
    il* corr_field;
    il* corr_index;
    il* best_corr_field = NULL;
    il* best_corr_index = NULL;
    double* sigma2s;

	assert(mo->wcs_valid || sip);

	// center and radius of the field in xyz space:
    fieldcenter = mo->center;
    fieldr2 = square(mo->radius);
    debug("Field center %g,%g,%g, radius2 %g\n", fieldcenter[0], fieldcenter[1], fieldcenter[2], fieldr2);

    // find index stars and project them into pixel coordinates.
    if (get_index_stars(fieldcenter, fieldr2, skdt, sip, &(mo->wcstan),
                        fieldW, fieldH, &indexpix, &starids, &NI)) {
		// I don't know HOW this happens - at the very least, the four stars
		// belonging to the quad that generated this hit should lie in the
		// proposed field - but I've seen it happen!
		mo->nfield = 0;
		mo->noverlap = 0;
		matchobj_compute_derived(mo);
		mo->logodds = -HUGE_VAL;
		return;
    }

    NF = starxy_n(vf->field);
	debug("Number of field stars: %i\n", NF);
	debug("Number of index stars: %i\n", NI);

	// If we're verifying an existing WCS solution, then don't increase the variance
	// away from the center of the matched quad.
    if (fake_match)
        do_gamma = FALSE;

    // If we're modelling the expected noise as a Gaussian whose variance grows
    // away from the quad center, compute the required quantities...
	if (do_gamma) {
        double Axy[2], Bxy[2];
        // Find the midpoint of AB of the quad in pixel space.
        starxy_get(vf->field, mo->field[0], Axy);
        starxy_get(vf->field, mo->field[1], Bxy);
        qc[0] = 0.5 * (Axy[0] + Bxy[0]);
        qc[1] = 0.5 * (Axy[1] + Bxy[1]);
        // Find the radius-squared of the quad = distsq(qc, A)
        rquad2 = distsq(Axy, qc, 2);
        debug("Quad radius = %g pixels\n", sqrt(rquad2));
	}
    logmsg("do_gamma: %s\n", (do_gamma ? "T" : "F"));
    logmsg("verify_pix2 = %g\n", verify_pix2);

    // Compute individual positional variances for every field star.
    sigma2s = malloc(NF * sizeof(double));
    for (i=0; i<NF; i++) {
        if (do_gamma) {
            double sxy[2];
            double R2, sigma2;
            starxy_get(vf->field, i, sxy);
            // Distance from the quad center of this field star:
            R2 = distsq(sxy, qc, 2);
            // Variance of a field star at that distance from the quad center:
            sigma2 = verify_pix2 * (1.0 + R2/rquad2);
            sigma2s[i] = sigma2;
        } else {
            sigma2s[i] = verify_pix2;
        }
    }


    // Reduce the number of index stars so that the "radius of relevance" is bigger
    // than the field.
    //trim_index_stars(&indexpix, &starids, &NI);


    // Deduplicate field stars based on positional variance and rank.
    bool* keepers = NULL;
    keepers = deduplicate_field_stars(vf, sigma2s);

    free(keepers);
    free(sigma2s);

    // Prime the array where we store conflicting-match info:
    // any match is an improvement, except for stars that form the matched quad.
	bestprob = malloc(NF * sizeof(double));
	for (i=0; i<NF; i++)
		bestprob[i] = -HUGE_VAL;
	// If we're verifying an existing WCS solution, then there is no match quad.
    if (!fake_match) {
        for (i=0; i<dimquads; i++) {
            assert(mo->field[i] >= 0);
            assert(mo->field[i] < NF);
            bestprob[mo->field[i]] = HUGE_VAL;
        }
    }

	// p(background) = 1/(W*H) of the image.
	logprob_background = -log(fieldW * fieldH);

	// p(distractor) = D / (W*H)
	logprob_distractor = log(distractors / (fieldW * fieldH));

	debug("log(p(background)) = %g\n", logprob_background);
	debug("log(p(distractor)) = %g\n", logprob_distractor);

    // add correspondences for the matched quad.
    corr_field = il_new(16);
    corr_index = il_new(16);
    if (!fake_match) {
        for (i=0; i<dimquads; i++) {
            il_append(corr_field, mo->field[i]);
            il_append(corr_index, mo->star[i]);
        }
    }

	bestlogodds = -HUGE_VAL;
	bestnmatch = bestnnomatch = bestnconflict = -1;
	nmatch = nnomatch = nconflict = 0;

	// Add index stars.
    for (i=0; i<NI; i++) {
        double bestd2;
        double sigma2;
        double R2 = 0.0;
        double logprob = -HUGE_VAL;
        int ind;
        int starid;
        int fldind;
        int j;
        bool cont;

        // Skip stars that are part of the quad:
        starid = starids[i];
        if (!fake_match) {
            cont = FALSE;
            for (j=0; j<dimquads; j++)
                if (starid == mo->star[j]) {
                    cont = TRUE;
                    break;
                }
            if (cont)
                continue;
        }

        // Find nearest field star.
        ind = kdtree_nearest_neighbour(vf->ftree, indexpix+i*2, &bestd2);
        fldind = vf->ftree->perm[ind];
        assert(fldind >= 0);
        assert(fldind < NF);

        if (do_gamma) {
            double sxy[2];
            starxy_get(vf->field, fldind, sxy);
            // Distance from the quad center of this field star:
            R2 = distsq(sxy, qc, 2);
            // Variance of a field star at that distance from the quad center:
            sigma2 = verify_pix2 * (1.0 + R2/rquad2);
        } else
            sigma2 = verify_pix2;

        if (DEBUGVERIFY) {
            if (do_gamma)
                debug("\nIndex star %i: rad %g pixels (%g quads) => sigma = %g.\n", i, sqrt(R2), sqrt(R2/rquad2), sqrt(sigma2));
            debug("Index star is at (%g,%g) pixels.\n", indexpix[i*2], indexpix[i*2+1]);
            debug("NN dist: %5.1f pix, %g sigmas\n", sqrt(bestd2), sqrt(bestd2/sigma2));
        }

        if (log((1.0 - distractors) / (2.0 * M_PI * sigma2 * NF)) < logprob_background) {
            debug("This Gaussian is nearly uninformative.\n");
            continue;
        }

        logprob = log((1.0 - distractors) / (2.0 * M_PI * sigma2 * NF)) - (bestd2 / (2.0 * sigma2));
        if (logprob < logprob_distractor) {
            debug("Distractor.\n");
            logprob = logprob_distractor;
            nnomatch++;
        } else {
            double oldprob;
            debug("Match (field star %i), logprob %g (background=%g, distractor=%g)\n", fldind, logprob, logprob_background, logprob_distractor);
            logprob = log(exp(logprob) + distractors/(fieldW*fieldH));

            oldprob = bestprob[fldind];
            if (oldprob != -HUGE_VAL) {
                nconflict++;
                // There was a previous match to this field object.
                if (oldprob == HUGE_VAL) {
                    debug("Conflicting match to one of the stars in the quad.\n");
                } else {
                    debug("Conflict: odds was %g, now %g.\n", oldprob, logprob);
                }
                // Allow an improved match (except to the stars composing the quad, because they are initialized to HUGE_VAL)
                if (logprob > oldprob) {
                    int ind;
                    oldprob = logprob;
                    logodds += (logprob - oldprob);
                    bestprob[fldind] = logprob;
                    // Replace the correspondence.
                    ind = il_index_of(corr_field, fldind);
                    il_set(corr_index, ind, starid);
                }
                continue;
            } else {
                bestprob[fldind] = logprob;
                nmatch++;
                il_append(corr_field, fldind);
                il_append(corr_index, starid);
            }
        }

        logodds += (logprob - logprob_background);
        debug("Logodds now %g\n", logodds);

        if (logodds < logratio_tobail)
            break;

		if (logodds > bestlogodds) {
			bestlogodds = logodds;
			bestnmatch = nmatch;
			bestnnomatch = nnomatch;
			bestnconflict = nconflict;

            if (best_corr_field)
                il_free(best_corr_field);
            if (best_corr_index)
                il_free(best_corr_index);
            best_corr_field = il_dupe(corr_field);
            best_corr_index = il_dupe(corr_index);
		}
	}

	free(bestprob);

	free(indexpix);
    free(starids);

    mo->corr_field = best_corr_field;
    mo->corr_index = best_corr_index;

    il_free(corr_field);
    il_free(corr_index);

	mo->logodds = bestlogodds;
	mo->noverlap = bestnmatch;
	mo->nconflict = bestnconflict;
	mo->nfield = bestnmatch + bestnnomatch + bestnconflict;
	mo->nindex = NI;
    matchobj_compute_derived(mo);

    if (mo->logodds > log(1e9)) {
        logverb("breakpoint ahoy!\n");
    }
}

