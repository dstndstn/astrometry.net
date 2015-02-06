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

#include "verify2.h"
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

#include "fitsioutils.h"
#include "errors.h"
#include "cairoutils.h"	

static void add_gaussian_to_image(double* img, int W, int H,
								  double cx, double cy, double sigma,
								  double scale,
								  double nsigma, int boundary) {
	int x, y;
	if (boundary == 0) {
		// truncate.
		for (y = MAX(0, cy - nsigma*sigma); y <= MIN(H-1, cy + nsigma * sigma); y++) {
			for (x = MAX(0, cx - nsigma*sigma); x <= MIN(W-1, cx + nsigma * sigma); x++) {
				img[y*W + x] += scale * exp(-(square(y-cy)+square(x-cx)) / (2.0 * square(sigma)));
			}
		}

	} else if (boundary == 1) {
		// mirror.
		int mx, my;
		for (y=MAX(-(H-1), floor(cy - nsigma * sigma));
			 y<=MIN(2*H-1, ceil(cy + nsigma * sigma)); y++) {
			if (y < 0)
				my = -1 - y;
			else if (y >= H)
				my = 2*H - 1 - y;
			else
				my = y;
			for (x=MAX(-(W-1), floor(cx - nsigma * sigma));
				 x<=MIN(2*W-1, ceil(cx + nsigma * sigma)); x++) {
				if (x < 0)
					mx = -1 - x;
				else if (x >= W)
					mx = 2*W - 1 - x;
				else
					mx = x;
				img[my*W + mx] += scale * exp(-(square(y-cy)+square(x-cx)) / (2.0 * square(sigma)));
			}
		}
	}
}

void verify_get_all_matches(const double* refxys, int NR,
							const double* testxys, const double* testsigma2s, int NT,
							double effective_area,
							double distractors,
							double nsigma,
							double limit,
							il*** p_reflist,
							dl*** p_problist) {
	double* refcopy;
	kdtree_t* rtree;
	int Nleaf = 10;
	int i,j;
	double logd;
	double logbg;
	double loglimit;

	il** reflist;
	dl** problist;

	reflist  = calloc(NT, sizeof(il*));
	problist = calloc(NT, sizeof(dl*));

	// Build a tree out of the index stars in pixel space...
	// kdtree scrambles the data array so make a copy first.
	refcopy = malloc(2 * NR * sizeof(double));
	memcpy(refcopy, refxys, 2 * NR * sizeof(double));
	rtree = kdtree_build(NULL, refcopy, NR, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);

	logbg = log(1.0 / effective_area);
	logd  = log(distractors / effective_area);
	loglimit = log(distractors / effective_area * limit);

	for (i=0; i<NT; i++) {
		const double* testxy;
		double sig2;
		kdtree_qres_t* res;

		testxy = testxys + 2*i;
		sig2 = testsigma2s[i];

		logverb("\n");
		logverb("test star %i: (%.1f,%.1f), sigma: %.1f\n", i, testxy[0], testxy[1], sqrt(sig2));

		// find all ref stars within nsigma.
		res = kdtree_rangesearch_options(rtree, testxy, sig2*nsigma*nsigma,
										 KD_OPTIONS_SORT_DISTS | KD_OPTIONS_SMALL_RADIUS);

		if (res->nres == 0) {
			kdtree_free_query(res);
			continue;
		}

		reflist[i] = il_new(4);
		problist[i] = dl_new(4);

		for (j=0; j<res->nres; j++) {
			double d2;
			int refi;
			double loggmax, logfg;

			d2 = res->sdists[j];
			refi = res->inds[j];

			// peak value of the Gaussian
			loggmax = log((1.0 - distractors) / (2.0 * M_PI * sig2 * NR));
			// value of the Gaussian
			logfg = loggmax - d2 / (2.0 * sig2);

			if (logfg < loglimit)
				continue;

			logverb("  ref star %i, dist %.2f, sigmas: %.3f, logfg: %.1f (%.1f above distractor, %.1f above bg, %.1f above keep-limit)\n",
					refi, sqrt(d2), sqrt(d2 / sig2), logfg, logfg - logd, logfg - logbg, logfg - loglimit);

			il_append(reflist[i], refi);
			dl_append(problist[i], logfg);
		}

		kdtree_free_query(res);
	}
	kdtree_free(rtree);
	free(refcopy);

	*p_reflist  = reflist;
	*p_problist = problist;
}


	/*
	unsigned char* img = malloc(4*W*H);

	// draw images of index and field densities.
	double* idensity = calloc(W * H, sizeof(double));
	double* fdensity = calloc(W * H, sizeof(double));

	double iscale = 2. * sqrt((double)(W * H) / (NI * M_PI));
	double fscale = 2. * sqrt((double)(W * H) / (NF * M_PI));
	logverb("NI = %i; iscale = %g\n", NI, iscale);
	logverb("NF = %i; fscale = %g\n", NF, fscale);
	logverb("computing density images...\n");
	for (i=0; i<NI; i++)
		add_gaussian_to_image(idensity, W, H, indexpix[i*2 + 0], indexpix[i*2 + 1], iscale, 1.0, 3.0, 1);
	for (i=0; i<NF; i++)
		add_gaussian_to_image(fdensity, W, H, starxy_getx(vf->field, i), starxy_gety(vf->field, i), fscale, 1.0, 3.0, 1);

	double idmax=0, fdmax=0;
	for (i=0; i<(W*H); i++) {
		idmax = MAX(idmax, idensity[i]);
		fdmax = MAX(fdmax, fdensity[i]);
	}
	for (i=0; i<(W*H); i++) {
		unsigned char val = 255.5 * idensity[i] / idmax;
		img[i*4+0] = val;
		img[i*4+1] = val;
		img[i*4+2] = val;
		img[i*4+3] = 255;
	}
	cairoutils_write_png("idensity.png", img, W, H);
	for (i=0; i<(W*H); i++) {
		unsigned char val = 255.5 * fdensity[i] / fdmax;
		img[i*4+0] = val;
		img[i*4+1] = val;
		img[i*4+2] = val;
		img[i*4+3] = 255;
	}
	cairoutils_write_png("fdensity.png", img, W, H);

	free(idensity);
	free(fdensity);
	 */


	/*
	 // index star weights.
	 double* iweights = malloc(NI * sizeof(double));
	 for (i=0; i<NI; i++)
	 iweights[i] = 1.0;
	 // don't count index stars that are part of the matched quad.
	 for (i=0; i<NI; i++)
	 for (j=0; j<dimquads; j++)
	 if (starids[i] == mo->star[j])
	 iweights[i] = 0.0;

	 double ranksigma = 20.0;
	 int Nrankprobs = (int)(ranksigma * 5);
	 double* rankprobs = malloc(Nrankprobs * sizeof(double));
	 for (i=0; i<Nrankprobs; i++)
	 rankprobs[i] = exp(-(double)(i*i) / (2. * ranksigma * ranksigma));

	 double qc[2], qr2;

	 get_quad_center(vf, mo, qc, &qr2);
	 */

	/*
		// create the probability distribution map for this field star.
		double* pmap = malloc(W * H * sizeof(double));
		// background rate...
		for (j=0; j<(W*H); j++)
			pmap[j] = distractors / (fieldW*fieldH);

		double normrankprob;
		int dr;
		normrankprob = 0.0;
		for (j=0; j<NI; j++) {
			dr = abs(i - j);
			if (dr >= Nrankprobs)
				continue;
			normrankprob += rankprobs[dr];
		}
		for (j=0; j<NI; j++) {
			double r2, sig2, sig;

			dr = abs(i - j);
			if (dr >= Nrankprobs)
				continue;

			r2 = distsq(indexpix + j*2, qc, 2);
			sig2 = get_sigma2_at_radius(verify_pix2, r2, qr2);
			sig = sqrt(sig2);

			for (y = MAX(0, indexpix[j*2 + 1] - 5.0*sig);
				 y <= MIN(H-1, indexpix[j*2 + 1] + 5.0 * sig); y++) {
				for (x = MAX(0, indexpix[j*2 + 0] - 5.0*sig);
					 x <= MIN(W-1, indexpix[j*2 + 0] + 5.0 * sig); x++) {
					pmap[y*W + x] += iweights[j] * (rankprobs[dr] / normrankprob) *
						exp(-(square(indexpix[j*2+0]-x) + square(indexpix[j*2+1]-y)) / (2.0 * sig2)) *
						(1.0 / (2.0 * M_PI * sig2)) * (1.0 - distractors);
				}
			}
		}

		double maxval = 0.0;
		for (j=0; j<(W*H); j++)
			maxval = MAX(maxval, pmap[j]);

		double psum = 0.0;
		for (j=0; j<(W*H); j++)
			psum += pmap[j];
		logverb("Probability sum: %g\n", psum);

		char* fn;

		printf("Writing probability image %i\n", i);
		printf("maxval = %g\n", maxval);

		for (j=0; j<(W*H); j++) {
			unsigned char val = (unsigned char)MAX(0, MIN(255, 255.5 * sqrt(pmap[j] / maxval)));
			img[4*j + 0] = val;
			img[4*j + 1] = val;
			img[4*j + 2] = val;
			img[4*j + 3] = 255;
		}

		free(pmap);

		cairo_t* cairo;
		cairo_surface_t* target;

		target = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, W, H, W*4);
		cairo = cairo_create(target);
		cairo_set_line_join(cairo, CAIRO_LINE_JOIN_ROUND);
		cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);

		// draw the quad.
		cairo_set_source_rgba(cairo, 0.0, 0.0, 1.0, 0.5);

		double starxy[DQMAX*2];
		double angles[DQMAX];
		int perm[DQMAX];
		int DQ = dimquads;

		for (k=0; k<DQ; k++)
            starxy_get(vf->field, mo->field[k], starxy + k*2);
		for (k=0; k<DQ; k++)
			angles[k] = atan2(starxy[k*2 + 1] - qc[1], starxy[k*2 + 0] - qc[0]);
		permutation_init(perm, DQ);
		permuted_sort(angles, sizeof(double), compare_doubles_asc, perm, DQ);
		for (k=0; k<DQ; k++) {
			double px, py;
			px = starxy[perm[k]*2+0];
			py = starxy[perm[k]*2+1];
			if (k == 0)
				cairo_move_to(cairo, px, py);
			else
				cairo_line_to(cairo, px, py);
		}
		cairo_close_path(cairo);
		cairo_stroke(cairo);

		// draw the source point.
		cairo_set_source_rgb(cairo, 1.0, 0.0, 0.0);
		cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CROSSHAIR, fxy[0], fxy[1], 3.0);
		cairo_stroke(cairo);

	 */


	/*
		// find index stars within n sigma...
		double nsigma = 5.0;
		double r2 = sigma2s[i] * nsigma*nsigma;
		int opts = KD_OPTIONS_COMPUTE_DISTS;

		kdtree_qres_t* res = kdtree_rangesearch_options(itree, fxy, r2, opts);

		logverb("found %i index stars within range.\n", res->nres);

		if (res->nres == 0)
			goto nextstar;

		double bestprob = 0.0;
		int bestarg = -1;

		for (j=0; j<res->nres; j++) {
			int ii = res->inds[j];
			dr = abs(i - ii);
			logverb("  index star %i: rank diff %i\n", ii, i-ii);
			if (dr >= Nrankprobs)
				continue;
			double prob = iweights[ii] * rankprobs[dr] *
				exp(-res->sdists[j] / (2.0 * sigma2s[i]));
			logverb("  -> prob %g\n", prob);
			if (prob > bestprob) {
				bestprob = prob;
				bestarg = j;
			}
		}

		if (bestarg == -1) {
			// FIXME -- distractor?
			goto nextstar;
		}

		int besti = res->inds[bestarg];
		double lostweight = bestprob / iweights[besti]; 
		// exp(-res->sdists[bestarg] / (2.0 * sigma2s[i]));
		iweights[besti] = MAX(0.0, iweights[besti] - lostweight);
		logverb("matched index star %i: dropping weight by %g; new weight %g.\n", besti, lostweight, iweights[besti]);
		
		bestprob *= (1.0 / (2.0 * M_PI * sigma2s[i])) / normrankprob * (1.0 - distractors);
		logverb("bestprob %g, vs background %g\n", bestprob, (1.0 / (double)(W * H)));
		
		kdtree_free_query(res);

		cairo_set_source_rgb(cairo, 0.0, 1.0, 0.0);
		cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE, indexpix[2*besti+0], indexpix[2*besti+1], 3.0);
		cairo_stroke(cairo);


	nextstar:
		cairoutils_argb32_to_rgba(img, W, H);
		cairo_surface_destroy(target);
		cairo_destroy(cairo);
		asprintf(&fn, "logprob-%04i.png", i);
		cairoutils_write_png(fn, img, W, H);
		free(img);


		if (i >= 9)
			break;
	}

	 kdtree_free(itree);
	 free(ipixcopy);

	 */







