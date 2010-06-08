/*
 This file is part of the Astrometry.net suite.
 Copyright 2010 Dustin Lang.

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

#include "bl.h"
#include "blind_wcs.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "scamp.h"
#include "log.h"
#include "errors.h"
#include "tweak.h"
#include "matchfile.h"
#include "matchobj.h"
#include "boilerplate.h"
#include "xylist.h"
#include "rdlist.h"
#include "mathutil.h"
#include "verify.h"
#include "plotstuff.h"
#include "plotimage.h"
#include "cairoutils.h"
#include "fitsioutils.h"

sip_t* tweak2(const double* fieldxy, int Nfield,
			  double fieldjitter,
			  int W, int H,
			  const double* indexradec, int Nindex,
			  double indexjitter,
			  const double* quadcenter, double quadR2,
			  double distractors,
			  double logodds_bail,
			  int sip_order,
			  const sip_t* startwcs,
			  sip_t* destwcs,
			  int** newtheta, double** newodds,
			  double* crpix) {
	int order;
	sip_t* sipout;
	int* indexin;
	double* indexpix;
	double* fieldsigma2s;
	double* weights;
	double* matchxyz;
	double* matchxy;
	int i, Nin=0;

	if (destwcs)
		sipout = destwcs;
	else
		sipout = sip_create();

	indexin = malloc(Nindex * sizeof(int));
	indexpix = malloc(2 * Nindex * sizeof(double));
	fieldsigma2s = malloc(Nfield * sizeof(double));
	weights = malloc(Nfield * sizeof(double));
	matchxyz = malloc(Nfield * 3 * sizeof(double));
	matchxy = malloc(Nfield * 2 * sizeof(double));

	// FIXME --- hmmm, how do the annealing steps and iterating up to
	// higher orders interact?

	assert(startwcs);
	memcpy(sipout, startwcs, sizeof(sip_t));

	if (!sipout->wcstan.imagew)
		sipout->wcstan.imagew = W;
	if (!sipout->wcstan.imageh)
		sipout->wcstan.imageh = H;

	for (order=1; order <= sip_order; order++) {
		int step;
		int STEPS = 20;
		// variance growth rate wrt radius.
		double gamma = 1.0;
		logverb("Starting tweak2 order=%i\n", order);

		for (step=0; step<STEPS; step++) {
			double iscale;
			double ijitter;
			double ra, dec;
			double R2, logodds;
			int besti, Nmatch;
			int* theta;
			double* odds;

			// Anneal
			gamma = pow(0.9, step);
			if (step == STEPS-1)
				gamma = 0.0;
			logverb("Annealing step %i, gamma = %g\n", step, gamma);
			
			logverb("Using input WCS:\n");
			sip_print_to(sipout, stdout);

			// FIXME --- this should be done in dstnthing, since it
			// isn't necessary when called during normal solving (and
			// it requires keeping the 'indexin' permutation).

			// Project RDLS into pixel space; keep the ones inside image bounds.
			Nin = 0;
			for (i=0; i<Nindex; i++) {
				bool ok;
				double x,y;
				ra  = indexradec[2*i + 0];
				dec = indexradec[2*i + 1];
				ok = sip_radec2pixelxy(sipout, ra, dec, &x, &y);
				if (!ok)
					continue;
				if (!sip_pixel_is_inside_image(sipout, x, y))
					continue;
				indexpix[Nin*2+0] = x;
				indexpix[Nin*2+1] = y;
				indexin[Nin] = i;
				Nin++;
			}
			logverb("%i reference sources within the image.\n", Nin);
			//logverb("CRPIX is (%g,%g)\n", sip.wcstan.crpix[0], sip.wcstan.crpix[1]);
			iscale = sip_pixel_scale(sipout);
			ijitter = indexjitter / iscale;
			logverb("With pixel scale of %g arcsec/pixel, index adds jitter of %g pix.\n", iscale, ijitter);

			for (i=0; i<Nfield; i++) {
				R2 = distsq(quadcenter, fieldxy + 2*i, 2);
				fieldsigma2s[i] = (square(fieldjitter) + square(ijitter)) * (1.0 + gamma * R2/quadR2);
			}

			logodds = verify_star_lists(indexpix, Nin,
										fieldxy, fieldsigma2s, Nfield,
										W*H, distractors,
										logodds_bail, HUGE_VAL,
										&besti, &odds, &theta, NULL);
			logverb("Logodds: %g\n", logodds);
			logverb("besti: %i\n", besti);

			logverb("  Hit/miss: ");
			if (log_get_level() >= LOG_VERB)
				verify_log_hit_miss(theta, NULL, besti+1, Nfield, LOG_VERB);
			logverb("\n");


			/*		if (plotfn) {
			 char fn[256];
			 sprintf(fn, "%s-%02i%c.png", plotfn, step, 'a');
			 makeplot(fn, bgimgfn, W, H, Nfield, fieldxy, fieldsigma2s,
			 Nin, indexpix, Nfield-1, theta, sip.wcstan.crpix);
			 }*/

			Nmatch = 0;
			logverb("Weights:");
			for (i=0; i<Nfield; i++) {
				double ra,dec;
				if (theta[i] < 0)
					continue;
				ra  = indexradec[indexin[theta[i]]*2+0];
				dec = indexradec[indexin[theta[i]]*2+1];
				radecdeg2xyzarr(ra, dec, matchxyz + Nmatch*3);
				memcpy(matchxy + Nmatch*2, fieldxy + i*2, 2*sizeof(double));
				weights[Nmatch] = verify_logodds_to_weight(odds[i]);
				logverb(" %.2f", weights[Nmatch]);
				Nmatch++;
			}
			logverb("\n");

			free(theta);
			free(odds);

			if (order == 1) {
				tan_t newtan;
				blind_wcs_compute_weighted(matchxyz, matchxy, weights, Nmatch, &newtan, NULL);
				newtan.imagew = W;
				newtan.imageh = H;
				sip_wrap_tan(&newtan, sipout);

				//logverb("Original TAN WCS:\n");
				//tan_print_to(&sip.wcstan, stdout);
				logverb("Using %i (weighted) matches, new TAN WCS is:\n", Nmatch);
				tan_print_to(&newtan, stdout);

				if (crpix) {
					tan_t temptan;
					logverb("Moving tangent point to given CRPIX (%g,%g)\n", crpix[0], crpix[1]);

					blind_wcs_move_tangent_point_weighted(matchxyz, matchxy, weights, Nmatch,
														  crpix, &newtan, &temptan);
					blind_wcs_move_tangent_point_weighted(matchxyz, matchxy, weights, Nmatch,
														  crpix, &temptan, &newtan);
					newtan.imagew = W;
					newtan.imageh = H;
					sip_wrap_tan(&newtan, sipout);
					logverb("After moving CRPIX, TAN WCS is:\n");
					tan_print_to(&newtan, stdout);
				}

				/*if (plotfn) {
				 char fn[256];
				 for (i=0; i<Nindex; i++) {
				 bool ok;
				 rd_getradec(rd, i, &ra, &dec);
				 ok = tan_radec2pixelxy(&newtan, ra, dec, indexpix + i*2, indexpix + i*2 + 1);
				 assert(ok);
				 }
				 sprintf(fn, "%s-%02i%c.png", plotfn, step, 'b');
				 makeplot(fn, bgimgfn, W, H, Nfield, fieldpix, fieldsigma2s,
				 Nindex, indexpix, Nfield-1, theta, newtan.crpix);
				 }*/
			} else {
				tweak_t* t = tweak_new();
				starxy_t* sxy = starxy_new(Nmatch, FALSE, FALSE);
				il* imginds = il_new(256);
				il* refinds = il_new(256);
				dl* wts = dl_new(256);

				for (i=0; i<Nmatch; i++) {
					starxy_set_x(sxy, i, matchxy[2*i+0]);
					starxy_set_y(sxy, i, matchxy[2*i+1]);
				}
				tweak_init(t);
				tweak_push_ref_xyz(t, matchxyz, Nmatch);
				tweak_push_image_xy(t, sxy);
				for (i=0; i<Nmatch; i++) {
					il_append(imginds, i);
					il_append(refinds, i);
					dl_append(wts, weights[i]);
				}
				tweak_push_correspondence_indices(t, imginds, refinds, NULL, wts);
				tweak_push_wcs_tan(t, &sipout->wcstan);
				t->sip->a_order = t->sip->b_order = t->sip->ap_order = t->sip->bp_order = order;
				t->weighted_fit = TRUE;
				// We don't really want to iterate, since tweak will do its own
				// correspondences in a bad way.
				//for (i=0; i<10; i++) {
				tweak_go_to(t, TWEAK_HAS_LINEAR_CD);
				//logverb("\n");
				//sip_print_to(t->sip, stdout);
				//t->state &= ~TWEAK_HAS_LINEAR_CD;
				//}
				logverb("Got SIP:\n");
				if (log_get_level() >= LOG_VERB)
					sip_print_to(t->sip, stdout);
				memcpy(sipout, t->sip, sizeof(sip_t));
				sipout->wcstan.imagew = W;
				sipout->wcstan.imageh = H;
				/*if (plotfn) {
				 char fn[256];
				 for (i=0; i<Nindex; i++) {
				 bool ok;
				 rd_getradec(rd, i, &ra, &dec);
				 ok = sip_radec2pixelxy(newsip, ra, dec, indexpix + i*2, indexpix + i*2 + 1);
				 assert(ok);
				 }
				 sprintf(fn, "%s-%02i%c.png", plotfn, step, 'c');
				 makeplot(fn, bgimgfn, W, H, Nfield, fieldpix, fieldsigma2s,
				 Nindex, indexpix, Nfield-1, theta, newsip->wcstan.crpix);
				 }*/
				starxy_free(sxy);
				tweak_free(t);
			}
		}
	}

	if (newtheta || newodds) {
		int besti;
		double logodds;
		logodds = verify_star_lists(indexpix, Nin,
									fieldxy, fieldsigma2s, Nfield,
									W*H, distractors,
									logodds_bail, HUGE_VAL,
									&besti, newodds, newtheta, NULL);
		logverb("Final logodds: %g\n", logodds);
		// undo the "indexpix" inside-image-bounds cut.
		for (i=0; i<=besti; i++) {
			if ((*newtheta)[i] < 0)
				continue;
			(*newtheta)[i] = indexin[(*newtheta)[i]];
		}

	}


	free(indexin);
	free(indexpix);
	free(fieldsigma2s);
	free(weights);
	free(matchxyz);
	free(matchxy);

	return sipout;
}

