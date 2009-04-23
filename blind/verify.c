/*
 This file is part of the Astrometry.net suite.
 Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
 Copyright 2009 Dustin Lang, David W. Hogg.

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
#include "sip-utils.h"
#include "healpix.h"

#define DEBUGVERIFY 1
#if DEBUGVERIFY
//#define debug(args...) fprintf(stderr, args)
#else
#define debug(args...)
#endif

/*
 This gets called once for each field before verification begins.  We
 build a kdtree out of the field stars (in pixel space) which will be
 used during deduplication.
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
    // Note on kdtree type: I tried U32 (duu) but it was marginally slower.
    // I didn't try U16 (dss) because we need a fair bit of accuracy here.
    // Make a copy of the field objects, because we're going to build a
    // kdtree out of them and that shuffles their order.
    vf->fieldcopy = starxy_copy_xy(fieldxy);
    vf->xy = starxy_copy_xy(fieldxy);
    if (!vf->fieldcopy || !vf->xy) {
        fprintf(stderr, "Failed to copy the field.\n");
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
	free(vf->xy);
    free(vf->fieldcopy);
    free(vf);
}

void verify_get_index_stars(const double* fieldcenter, double fieldr2,
							const startree_t* skdt, const sip_t* sip, const tan_t* tan,
							double fieldW, double fieldH,
							double** p_indexradec,
							double** indexpix, int** p_starids, int* p_nindex) {
	double* indxyz;
    int i, N, NI;
    int* sweep;
    int* starid;
	int* inbounds;
	int* perm;
	double* radec = NULL;

	assert(skdt->sweep);
	assert(p_nindex);
	assert(sip || tan);

	// Find all index stars within the bounding circle of the field.
	startree_search_for(skdt, fieldcenter, fieldr2, &indxyz, NULL, &starid, &N);

	if (!indxyz) {
		// no stars in range.
		*p_nindex = 0;
		return;
	}

	// Find index stars within the rectangular field.
	inbounds = sip_filter_stars_in_field(sip, tan, indxyz, NULL, N, indexpix,
										 NULL, &NI);
	// Apply the permutation now, so that "indexpix" and "starid" stay in sync:
	// indexpix is already in the "inbounds" ordering.
	permutation_apply(inbounds, NI, starid, starid, sizeof(int));

	// Compute index RA,Decs if requested.
	if (p_indexradec) {
		radec = malloc(2 * NI * sizeof(double));
		for (i=0; i<NI; i++)
			// note that the "inbounds" permutation is applied to "indxyz" here.
			// we will apply the sweep permutation below.
			xyzarr2radecdegarr(indxyz + 3*inbounds[i], radec + 2*i);
		*p_indexradec = radec;
	}
	free(indxyz);
	free(inbounds);

    // Each index star has a "sweep number" assigned during index building;
    // it roughly represents a local brightness ordering.  Use this to sort the
	// index stars.
	sweep = malloc(NI * sizeof(int));
	for (i=0; i<NI; i++)
		sweep[i] = skdt->sweep[starid[i]];
    perm = permuted_sort(sweep, sizeof(int), compare_ints_asc, NULL, NI);
	free(sweep);

	if (indexpix) {
		permutation_apply(perm, NI, *indexpix, *indexpix, 2 * sizeof(double));
		*indexpix = realloc(*indexpix, NI * 2 * sizeof(double));
	}

	if (p_starids) {
		permutation_apply(perm, NI, starid, starid, sizeof(int));
		starid = realloc(starid, NI * sizeof(int));
		*p_starids = starid;
	} else
		free(starid);

	if (p_indexradec)
		permutation_apply(perm, NI, radec, radec, 2 * sizeof(double));

	free(perm);

    *p_nindex = NI;
}

//static void trim_index_stars()

/**
 If field objects are within "sigma" of each other (where sigma depends on the
 distance from the matched quad), then they are not very useful for verification.
 We filter out field stars within sigma of each other, taking only the brightest.

 Returns an array indicating which field stars should be kept.
 */
static bool* deduplicate_field_stars(verify_field_t* vf, double* sigma2s, double nsigmas) {
    bool* keepers = NULL;
    int i, j, N;
    kdtree_qres_t* res;
	double nsig2 = nsigmas*nsigmas;

    N = starxy_n(vf->field);
    keepers = malloc(N * sizeof(bool));
    for (i=0; i<N; i++)
		keepers[i] = TRUE;
    for (i=0; i<N; i++) {
        double sxy[2];
		if (!keepers[i])
			continue;
        starxy_get(vf->field, i, sxy);
        res = kdtree_rangesearch_nosort(vf->ftree, sxy, nsig2 * sigma2s[i]);
        for (j=0; j<res->nres; j++) {
			int ind = res->inds[j];
            if (ind > i) {
                keepers[ind] = FALSE;
                // DEBUG
                /*
                 double otherxy[2];
                 starxy_get(vf->field, ind, otherxy);
                 logverb("Field star %i at %g,%g: is close to field star %i at %g,%g.  dist is %g, sigma is %g\n",
                 i, sxy[0], sxy[1], ind, otherxy[0], otherxy[1],
                 sqrt(distsq(sxy, otherxy, 2)), sqrt(nsig2 * sigma2s[i]));
                 */
            }
        }
        kdtree_free_query(res);
    }
    return keepers;
}


static double get_sigma2_at_radius(double verify_pix2, double r2, double quadr2) {
	return verify_pix2 * (1.0 + r2/quadr2);
}

static double* compute_sigma2s(const verify_field_t* vf,
							   const double* xy, int NF,
							   const double* qc, double Q2,
							   double verify_pix2, bool do_gamma) {
	double* sigma2s;
    int i;
	double R2;

    sigma2s = malloc(NF * sizeof(double));
	if (!do_gamma) {
		for (i=0; i<NF; i++)
            sigma2s[i] = verify_pix2;
	} else {
		// Compute individual positional variances for every field
		// star.
		for (i=0; i<NF; i++) {
			if (vf) {
				double sxy[2];
				starxy_get(vf->field, i, sxy);
				// Distance from the quad center of this field star:
				R2 = distsq(sxy, qc, 2);
			} else
				R2 = distsq(xy + 2*i, qc, 2);

            // Variance of a field star at that distance from the quad center:
            sigma2s[i] = get_sigma2_at_radius(verify_pix2, R2, Q2);
        }
	}
	return sigma2s;
}

static void get_quad_center(const verify_field_t* vf, const MatchObj* mo, double* centerpix,
							double* quadr2) {
	double Axy[2], Bxy[2];
	// Find the midpoint of AB of the quad in pixel space.
	starxy_get(vf->field, mo->field[0], Axy);
	starxy_get(vf->field, mo->field[1], Bxy);
	centerpix[0] = 0.5 * (Axy[0] + Bxy[0]);
	centerpix[1] = 0.5 * (Axy[1] + Bxy[1]);
	// Find the radius-squared of the quad = distsq(qc, A)
	*quadr2 = distsq(Axy, centerpix, 2);
}

double* verify_compute_sigma2s(const verify_field_t* vf, const MatchObj* mo,
							   double verify_pix2, bool do_gamma) {
	int NF;
	double qc[2];
	double Q2;
	NF = starxy_n(vf->field);
	if (do_gamma) {
		get_quad_center(vf, mo, qc, &Q2);
		debug("Quad radius = %g pixels\n", sqrt(Q2));
	}
	return compute_sigma2s(vf, NULL, NF, qc, Q2, verify_pix2, do_gamma);
}

double* verify_compute_sigma2s_arr(const double* xy, int NF,
								   const double* qc, double Q2,
								   double verify_pix2, bool do_gamma) {
	return compute_sigma2s(NULL, xy, NF, qc, Q2, verify_pix2, do_gamma);
}

static int get_xy_bin(const double* xy,
					  double fieldW, double fieldH,
					  int nw, int nh) {
	int ix, iy;
	ix = (int)floor(nw * xy[0] / fieldW);
	ix = MAX(0, MIN(nw-1, ix));
	iy = (int)floor(nh * xy[1] / fieldH);
	iy = MAX(0, MIN(nh-1, iy));
	return iy * nw + ix;
}

void verify_get_uniformize_scale(int cutnside, double scale, int W, int H, int* cutnw, int* cutnh) {
	double cutarcsec, cutpix;
	cutarcsec = healpix_side_length_arcmin(cutnside) * 60.0;
	cutpix = cutarcsec / scale;
	logverb("cut nside: %i\n", cutnside);
	logverb("cut scale: %g arcsec\n", cutarcsec);
	logverb("match scale: %g arcsec/pix\n", scale);
	logverb("cut scale: %g pixels\n", cutpix);
	if (cutnw)
		*cutnw = MAX(1, (int)round(W / cutpix));
	if (cutnh)
		*cutnh = MAX(1, (int)round(H / cutpix));
}

static void uniformize(const double* xy,
					   int* perm,
					   int N,
					   double fieldW, double fieldH,
					   int nw, int nh,
					   int** p_bincounts,
					   double** p_bincenters,
					   int** p_binids) {
	il** lists;
	int i,j,k,p;
	int* bincounts = NULL;
	int* binids = NULL;

	if (p_bincenters) {
		double* bxy = malloc(nw * nh * 2 * sizeof(double));
		for (j=0; j<nh; j++)
			for (i=0; i<nw; i++) {
				bxy[(j * nw + i)*2 +0] = (i + 0.5) * fieldW / (double)nw;
				bxy[(j * nw + i)*2 +1] = (j + 0.5) * fieldH / (double)nh;
			}
		*p_bincenters = bxy;
	}

	if (p_binids) {
		binids = malloc(N * sizeof(int));
		*p_binids = binids;
	}

	lists = malloc(nw * nh * sizeof(il*));
	for (i=0; i<(nw*nh); i++)
		lists[i] = il_new(16);

	// put the stars in the appropriate bins.
	for (i=0; i<N; i++) {
		int ind;
		int bin;
		ind = perm[i];
		bin = get_xy_bin(xy + 2*ind, fieldW, fieldH, nw, nh);
		il_append(lists[bin], ind);
	}

	if (p_bincounts) {
		// note the bin occupancies.
		bincounts = malloc(nw * nh * sizeof(int));
		for (i=0; i<(nw*nh); i++) {
			bincounts[i] = il_size(lists[i]);
			//logverb("bin %i has %i stars\n", i, bincounts[i]);
		}
		*p_bincounts = bincounts;
	}

	// make sweeps through the bins, grabbing one star from each.
	p=0;
	for (k=0;; k++) {
		for (j=0; j<nh; j++) {
			for (i=0; i<nw; i++) {
				int binid = j*nw + i;
				il* lst = lists[binid];
				if (k >= il_size(lst))
					continue;
				perm[p] = il_get(lst, k);
				if (binids)
					binids[p] = binid;
				p++;
			}
		}
		if (p == N)
			break;
	}
	assert(p == N);

	for (i=0; i<(nw*nh); i++)
		il_free(lists[i]);
	free(lists);
}


void verify_hit(startree_t* skdt, int index_cutnside, MatchObj* mo, sip_t* sip, verify_field_t* vf,
                double pix2, double distractors,
                double fieldW, double fieldH,
                double logbail, double logaccept, double logstoplooking,
                bool do_gamma, int dimquads, bool fake_match) {
	int i,j,k;
	double* fieldcenter;
	double fieldr2;
	// number of reference stars
	int NR;
	double* refxy;
    int* starids;
    int NT;
	bool* keepers = NULL;
    double* sigma2s;
	int uni_nw, uni_nh;
	int* perm;
	double effA;
	double qc[2], Q2;
	double* testxy;
	double K;
	double worst;
	int besti;
	int* theta;

	assert(mo->wcs_valid || sip);

	// center and radius of the field in xyz space:
    fieldcenter = mo->center;
    fieldr2 = square(mo->radius);
    debug("Field center %g,%g,%g, radius2 %g\n", fieldcenter[0], fieldcenter[1], fieldcenter[2], fieldr2);

    // find index stars and project them into pixel coordinates.
    verify_get_index_stars(fieldcenter, fieldr2, skdt, sip, &(mo->wcstan),
						   fieldW, fieldH, NULL, &refxy, &starids, &NR);
	if (!NR) {
		// I don't know HOW this happens - at the very least, the four stars
		// belonging to the quad that generated this hit should lie in the
		// proposed field - but I've seen it happen!
		mo->nfield = 0;
		mo->nmatch = 0;
		matchobj_compute_derived(mo);
		mo->logodds = -HUGE_VAL;
		return;
    }

	// remove reference stars that are part of the quad.
	k = 0;
	if (!fake_match) {
		for (i=0; i<NR; i++) {
			bool inquad = FALSE;
			for (j=0; j<dimquads; j++)
				if (starids[i] == mo->star[j]) {
					inquad = TRUE;
					logverb("Skipping ref star index %i, starid %i: quad star %i.  k=%i\n",
							i, starids[i], j, k);
					break;
				}
			if (inquad)
				continue;
			if (i != k) {
				memcpy(refxy + 2*k, refxy + 2*i, 2*sizeof(double));
				starids[k] = starids[i];
			}
			k++;
		}
		// DEBUG
		/* Hmm, this happens occasionally...
		 if (k != NR - dimquads) {
		 double xyz[3];
		 double px,py;
		 logmsg("W=%g, H=%g\n", fieldW, fieldH);
		 for (j=0; j<dimquads; j++) {
		 startree_get(skdt, mo->star[j], xyz);
		 tan_xyzarr2pixelxy(&mo->wcstan, xyz, &px, &py);
		 logmsg("Quad star %i, id %i: pixel (%g,%g)\n", j, mo->star[j], px, py);
		 }
		 }
		 assert(k == NR - dimquads);
		 */
		NR = k;
	}

    NT = starxy_n(vf->field);
	debug("Number of test stars: %i\n", NT);
	debug("Number of reference stars: %i\n", NR);

	// If we're verifying an existing WCS solution, then don't increase the variance
	// away from the center of the matched quad.
    if (fake_match)
        do_gamma = FALSE;

	sigma2s = verify_compute_sigma2s(vf, mo, pix2, do_gamma);

	if (!fake_match)
		get_quad_center(vf, mo, qc, &Q2);

	// Deduplicate test stars.  This could be done (approximately) in preprocessing.
	keepers = deduplicate_field_stars(vf, sigma2s, 1.0);

	// Remove test quad stars.  Do this after deduplication so we
	// don't end up with test stars near the quad stars.
    if (!fake_match) {
		for (i=0; i<dimquads; i++) {
            assert(mo->field[i] >= 0);
            assert(mo->field[i] < NT);
            keepers[mo->field[i]] = FALSE;
		}
	}

	// Uniformize test stars
	// FIXME - can do this (possibly at several scales) in preprocessing.

	// -first apply the deduplication and removal of test stars by computing an
	// initial permutation array.
	perm = malloc(NT * sizeof(int));
	k = 0;
	for (i=0; i<NT; i++) {
		if (!keepers[i])
			continue;
		perm[k] = i;
		k++;
	}
	NT = k;
	free(keepers);

	// -get uniformization scale.
	verify_get_uniformize_scale(index_cutnside, mo->scale, fieldW, fieldH, &uni_nw, &uni_nh);
	logverb("uniformizing into %i x %i blocks.\n", uni_nw, uni_nh);

	// uniformize!
	if (uni_nw > 1 || uni_nh > 1) {
		double* bincenters;
		int* binids;
		double ror2;
		bool* goodbins = NULL;
		int Ngoodbins;

		uniformize(vf->xy, perm, NT, fieldW, fieldH, uni_nw, uni_nh, NULL, &bincenters, &binids);

		ror2 = Q2 * (1 + fieldW*fieldH*(1 - distractors) / (2. * M_PI * NR * pix2));
		logverb("Radius of relevance is %.1f\n", sqrt(ror2));
		goodbins = malloc(uni_nw * uni_nh * sizeof(bool));
		Ngoodbins = 0;
		for (i=0; i<(uni_nw * uni_nh); i++) {
			double binr2 = distsq(bincenters + 2*i, qc, 2);
			goodbins[i] = (binr2 < ror2);
			if (goodbins[i])
				Ngoodbins++;
		}
		// Remove test stars in irrelevant bins...
		k = 0;
		for (i=0; i<NT; i++) {
			if (!goodbins[binids[i]])
				continue;
			perm[k] = perm[i];
			k++;
		}
		NT = k;
		logverb("After removing %i/%i irrelevant bins: %i test stars.\n", (uni_nw*uni_nh)-Ngoodbins, uni_nw*uni_nh, NT);

		// Effective area: A * proportion of good bins.
		effA = fieldW * fieldH * Ngoodbins / (double)(uni_nw * uni_nh);

		// Remove reference stars in bad bins.
		k = 0;
		for (i=0; i<NR; i++) {
			int binid = get_xy_bin(refxy + 2*i, fieldW, fieldH, uni_nw, uni_nh);
			if (!goodbins[binid])
				continue;
			if (i != k)
				memcpy(refxy + 2*k, refxy + 2*i, 2*sizeof(double));
			k++;
		}
		NR = k;
		logverb("After removing irrelevant ref stars: %i ref stars.\n", NR);

		// New ROR is...
		logverb("ROR changed from %g to %g\n", sqrt(ror2),
				sqrt(Q2 * (1 + effA*(1 - distractors) / (2. * M_PI * NR * pix2))));

		free(goodbins);
		free(bincenters);
		free(binids);
	} else {
		effA = fieldW * fieldH;
	}


	testxy = malloc(NT * 2 * sizeof(double));

	permutation_apply(perm, NT, vf->xy, testxy, 2*sizeof(double));
	permutation_apply(perm, NT, sigma2s, sigma2s, sizeof(double));

	free(perm);

	K = verify_star_lists(refxy, NR, testxy, sigma2s, NT, effA, distractors,
						  logbail, logstoplooking, &besti, NULL, &theta, &worst);

	// FIXME - mo->corr_*, mo->nmatch, nnomatch, nconflict.
	mo->logodds = K;
	mo->worstlogodds = worst;
	mo->nfield = NT;
	mo->nindex = NR;

	if (K >= logaccept) {
		mo->nmatch = 0;
		mo->nconflict = 0;
		mo->ndistractor = 0;
		for (i=0; i<=besti; i++) {
			if (theta[i] == THETA_DISTRACTOR)
				mo->ndistractor++;
			else if (theta[i] == THETA_CONFLICT)
				mo->nconflict++;
			else
				mo->nmatch++;
		}
		matchobj_compute_derived(mo);
	}

	free(theta);
	free(testxy);
    free(sigma2s);
	free(refxy);
    free(starids);
}

static double logd_at(double distractor, int mu, int NR, double logbg) {
	return log(distractor + (1.0-distractor)*mu / (double)NR) + logbg;
}

double verify_star_lists(const double* refxys, int NR,
						 const double* testxys, const double* testsigma2s, int NT,
						 double effective_area,
						 double distractors,
						 double logodds_bail,
						 double logodds_stoplooking,
						 int* p_besti,
						 double** p_all_logodds, int** p_theta,
						 double* p_worstlogodds) {
	int i, j;
	double worstlogodds;
	double bestworstlogodds;
	double bestlogodds;
	int besti;
	double logodds;
	double logbg;
	double logd;
	//double matchnsigma = 5.0;
	double* refcopy;
	kdtree_t* rtree;
	int Nleaf = 10;
	int* rmatches;
	double* rprobs;
	double* all_logodds = NULL;
	int* theta = NULL;
	int mu;

	// Build a tree out of the index stars in pixel space...
	// kdtree scrambles the data array so make a copy first.
	refcopy = malloc(2 * NR * sizeof(double));
	memcpy(refcopy, refxys, 2 * NR * sizeof(double));
	rtree = kdtree_build(NULL, refcopy, NR, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);

	rmatches = malloc(NR * sizeof(int));
	for (i=0; i<NR; i++)
		rmatches[i] = -1;

	rprobs = malloc(NR * sizeof(double));
	for (i=0; i<NR; i++)
		rprobs[i] = -HUGE_VAL;

	if (p_all_logodds) {
		all_logodds = calloc(NT, sizeof(double));
		*p_all_logodds = all_logodds;
	}

	theta = malloc(NT * sizeof(int));

	logbg = log(1.0 / effective_area);

	worstlogodds = HUGE_VAL;
	bestlogodds = -HUGE_VAL;
	besti = -1;

	logodds = 0.0;
	mu = 0;
	for (i=0; i<NT; i++) {
		const double* testxy;
		double sig2;
		int refi;
		int tmpi;
		double d2;
		double logfg;

		testxy = testxys + 2*i;
		sig2 = testsigma2s[i];

		logd = logd_at(distractors, mu, NR, logbg);

		logverb("\n");
		logverb("test star %i: (%.1f,%.1f), sigma: %.1f\n", i, testxy[0], testxy[1], sqrt(sig2));

		// find nearest ref star (within 5 sigma)
        tmpi = kdtree_nearest_neighbour_within(rtree, testxy, sig2 * 25.0, &d2);
		if (tmpi == -1) {
			// no nearest neighbour within range.
			logverb("  No nearest neighbour.\n");
			refi = -1;
			logfg = -HUGE_VAL;
		} else {
			double loggmax;

			refi = kdtree_permute(rtree, tmpi);

			// peak value of the Gaussian
			loggmax = log((1.0 - distractors) / (2.0 * M_PI * sig2 * NR));

			// FIXME - uninformative?
			if (loggmax < logbg)
				logverb("  This star is uninformative: peak %.1f, bg %.1f.\n", loggmax, logbg);

			// value of the Gaussian
			logfg = loggmax - d2 / (2.0 * sig2);
			
			logverb("  NN: ref star %i, dist %.2f, sigmas: %.3f, logfg: %.1f (%.1f above distractor, %.1f above bg)\n",
					refi, sqrt(d2), sqrt(d2 / sig2), logfg, logfg - logd, logfg - logbg);
		}

		if (logfg < logd) {
			logfg = logd;
			logverb("  Distractor.\n");
			theta[i] = THETA_DISTRACTOR;

		} else {
			// duplicate match?
			if (rmatches[refi] != -1) {
				double oldfg = rprobs[refi];

				//logverb("Conflict: odds was %g, now %g.\n", oldfg, logfg);

				// Conflict.  Compute probabilities of old vs new theta.
				// keep the old one: the new star is a distractor
				double keepfg = logd;

				// switch to the new one: the new star is a match...
				double switchfg = logfg;
				// ... and the old one becomes a distractor...
				int oldj = rmatches[refi];
				int muj = 0;
				for (j=0; j<oldj; j++)
					if (theta[j] >= 0)
						muj++;
				switchfg += (logd_at(distractors, muj, NR, logbg) - oldfg);

				// FIXME - could estimate/bound the distractor change and avoid computing it...

				// ... and the intervening distractors become worse.
				logverb("  oldj is %i, muj is %i.\n", oldj, muj);
				logverb("  changing old point to distractor: %.1f change in logodds\n",
						(logd_at(distractors, muj, NR, logbg) - oldfg));
				for (; j<i; j++)
					if (theta[j] < 0) {
						switchfg += (logd_at(distractors, muj, NR, logbg) -
									 logd_at(distractors, muj+1, NR, logbg));
						logverb("  adjusting distractor %i: %g change in logodds\n",
								j, (logd_at(distractors, muj, NR, logbg) -
									logd_at(distractors, muj+1, NR, logbg)));
					} else
						muj++;
				logverb("  Conflict: keeping   old match, logfg would be %.1f\n", keepfg);
				logverb("  Conflict: accepting new match, logfg would be %.1f\n", switchfg);
				
				if (switchfg > keepfg) {
					// upgrade: old match becomes a distractor.
					logverb("  Conflict: upgrading.\n");
					//logodds += (logd - oldfg);
					//logverb("  Switching old match to distractor: logodds change %.1f, now %.1f\n",
					//(logd - oldfg), logodds);

					theta[oldj] = THETA_CONFLICT;
					theta[i] = refi;
					// record this new match.
					rmatches[refi] = i;
					rprobs[refi] = logfg;

					// "switchfg" incorporates the cost of adjusting the previous probabilities.
					logfg = switchfg;

				} else {
					// old match was better: this match becomes a distractor.
					logverb("  Conflict: not upgrading.\n"); //  logprob was %.1f, now %.1f.\n", oldfg, logfg);
					logfg = keepfg;
					theta[i] = THETA_CONFLICT;
				}
				// no change in mu.


			} else {
				// new match.
				rmatches[refi] = i;
				rprobs[refi] = logfg;
				theta[i] = refi;
				mu++;
			}
		}

        logodds += (logfg - logbg);
        logverb("  Logodds: change %.1f, now %.1f\n", (logfg - logbg), logodds);

		if (all_logodds)
			all_logodds[i] = logodds;

        if (logodds < logodds_bail) {
			if (all_logodds)
				for (j=i+1; j<NT; j++)
					all_logodds[j] = logodds;
            break;
		}

		worstlogodds = MIN(worstlogodds, logodds);

        if (logodds > bestlogodds) {
			bestlogodds = logodds;
			besti = i;
			// Record the worst log-odds we've seen up to this point.
			bestworstlogodds = worstlogodds;
		}

		if (logodds > logodds_stoplooking)
			break;
	}

	free(rmatches);

	if (p_theta)
		*p_theta = theta;
	else
		free(theta);

	if (p_besti)
		*p_besti = besti;

	if (p_worstlogodds)
		*p_worstlogodds = bestworstlogodds;

	free(rprobs);

	kdtree_free(rtree);
	free(refcopy);

	return bestlogodds;
}
