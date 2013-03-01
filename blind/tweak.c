/*
 This file is part of the Astrometry.net suite.
 Copyright 2006, 2007 Keir Mierle, David W. Hogg, Sam Roweis and Dustin Lang.
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
#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

#include "sip-utils.h"
#include "tweak.h"
#include "healpix.h"
#include "dualtree_rangesearch.h"
#include "kdtree_fits_io.h"
#include "mathutil.h"
#include "log.h"
#include "permutedsort.h"
#include "gslutils.h"
#include "errors.h"

// TODO:
//
//  1. Write the document which explains every step of tweak in detail with
//     comments relating exactly to the code.
//  2. Implement polynomial terms - use order parameter to zero out A matrix
//  3. Make it robust to outliers
//     - Make jitter evolve as fit evolves
//     - Sigma clipping
//  4. Need test image with non-trivial rotation to test CD transpose problem
//
//
//  - Put CD inverse into its own function
//
//  BUG? transpose of CD matrix is similar to CD matrix!
//  BUG? inverse when computing sx/sy (i.e. same transpose issue)
//  Ability to fit without re-doing correspondences
//  Split fit x/y (i.e. two fits one for x one for y)

#define KERNEL_SIZE 5
#define KERNEL_MARG ((KERNEL_SIZE-1)/2)

sip_t* tweak_just_do_it(const tan_t* wcs, const starxy_t* imagexy,
                        const double* starxyz,
                        const double* star_ra, const double* star_dec,
                        const double* star_radec,
                        int nstars, double jitter_arcsec,
                        int order, int inverse_order, int iterations,
                        anbool weighted, anbool skip_shift) {
	tweak_t* twee = NULL;
	sip_t* sip = NULL;

	twee = tweak_new();
    twee->jitter = jitter_arcsec;
	twee->sip->a_order  = twee->sip->b_order  = order;
	twee->sip->ap_order = twee->sip->bp_order = inverse_order;
	twee->weighted_fit = weighted;
    if (skip_shift)
		tweak_skip_shift(twee);

    tweak_push_image_xy(twee, imagexy);
    if (starxyz)
        tweak_push_ref_xyz(twee, starxyz, nstars);
    else if (star_ra && star_dec)
        tweak_push_ref_ad(twee, star_ra, star_dec, nstars);
    else if (star_radec)
        tweak_push_ref_ad_array(twee, star_radec, nstars);
    else {
        logerr("Need starxyz, (star_ra and star_dec), or star_radec");
        return NULL;
    }
	tweak_push_wcs_tan(twee, wcs);
    tweak_iterate_to_order(twee, order, iterations);

	// Steal the resulting SIP structure
	sip = twee->sip;
	twee->sip = NULL;

	tweak_free(twee);
	return sip;
}

static void get_dydx_range(double* ximg, double* yimg, int nimg,
                           double* xcat, double* ycat, int ncat,
                           double *mindx, double *mindy,
                           double *maxdx, double *maxdy) {
	int i, j;
	*maxdx = -1e100;
	*mindx = 1e100;
	*maxdy = -1e100;
	*mindy = 1e100;

	for (i = 0; i < nimg; i++) {
		for (j = 0; j < ncat; j++) {
			double dx = ximg[i] - xcat[j];
			double dy = yimg[i] - ycat[j];
			*maxdx = MAX(dx, *maxdx);
			*maxdy = MAX(dy, *maxdy);
			*mindx = MIN(dx, *mindx);
			*mindy = MIN(dy, *mindy);
		}
	}
}

static void get_shift(double* ximg, double* yimg, int nimg,
                      double* xcat, double* ycat, int ncat,
                      double mindx, double mindy, double maxdx, double maxdy,
                      double* xshift, double* yshift) {
	int i, j;
	int themax, themaxind, ys, xs;

	// hough transform
	int hsz = 1000; // hough histogram size (per side)
	int *hough = calloc(hsz * hsz, sizeof(int)); // allocate bins
	int kern[] = {0, 2, 3, 2, 0,  // approximate gaussian smoother
	              2, 7, 12, 7, 2,       // should be KERNEL_SIZE x KERNEL_SIZE
	              3, 12, 20, 12, 3,
	              2, 7, 12, 7, 2,
	              0, 2, 3, 2, 0};

    assert(sizeof(kern) == KERNEL_SIZE * KERNEL_SIZE * sizeof(int));

	for (i = 0; i < nimg; i++) {    // loop over all pairs of source-catalog objs
		for (j = 0; j < ncat; j++) {
			double dx = ximg[i] - xcat[j];
			double dy = yimg[i] - ycat[j];
			int hszi = hsz - 1;
			int iy = hszi * ( (dy - mindy) / (maxdy - mindy) ); // compute deltay using implicit floor
			int ix = hszi * ( (dx - mindx) / (maxdx - mindx) ); // compute deltax using implicit floor

			// check to make sure the point is in the box
			if (KERNEL_MARG <= iy && iy < hsz - KERNEL_MARG &&
                KERNEL_MARG <= ix && ix < hsz - KERNEL_MARG) {
				int kx, ky;
				for (ky = -2; ky <= 2; ky++)
					for (kx = -2; kx <= 2; kx++)
						hough[(iy - ky)*hsz + (ix - kx)] += kern[(ky + 2) * 5 + (kx + 2)];
			}
		}
	}

    // find argmax in hough
	themax = 0;
	themaxind = -1;
	for (i = 0; i < hsz*hsz; i++) {
		if (themax < hough[i]) {
			themaxind = i;
			themax = hough[i];
		}
	}
    // which hough bin is the max?
	ys = themaxind / hsz;
	xs = themaxind % hsz;


	*yshift = (ys / (double)hsz) * (maxdy - mindy) + mindy;
	*xshift = (xs / (double)hsz) * (maxdx - mindx) + mindx;
	debug("xs = %d, ys = %d\n", xs, ys);
	debug("get_shift: mindx=%g, maxdx=%g, mindy=%g, maxdy=%g\n", mindx, maxdx, mindy, maxdy);
	debug("get_shift: xs=%g, ys=%g\n", *xshift, *yshift);

	free(hough);
}

// Take shift in image plane and do a switcharoo to make the wcs something better
// in other words, take the shift in pixels and reset the WCS (in WCS coords)
// so that the new pixel shift would be zero
// FIXME -- dstn says, why not just
// sip_pixelxy2radec(wcs, crpix0 +- xs, crpix1 +- ys, wcs->wcstan.crval+0, wcs->wcstan.crval+1);
sip_t* wcs_shift(sip_t* wcs, double xs, double ys) {
	// UNITS: crpix and xs/ys in pixels, crvals in degrees, nx/nyref and theta in degrees
	double crpix0, crpix1, crval0;
	double nxref, nyref, theta, sintheta, costheta;
	double newCD[2][2]; //the new CD matrix
	sip_t* swcs = malloc(sizeof(sip_t));
	memcpy(swcs, wcs, sizeof(sip_t));

	// Save old vals
	crpix0 = wcs->wcstan.crpix[0];
	crpix1 = wcs->wcstan.crpix[1];
	crval0 = wcs->wcstan.crval[0];

    // compute the desired projection of the new tangent point by
    // shifting the projection of the current tangent point
	wcs->wcstan.crpix[0] += xs;
	wcs->wcstan.crpix[1] += ys;

	//logmsg("wcs_shift: xs,ys=(%g,%g)\n", xs, ys);

	// now reproject the old crpix[xy] into shifted wcs
	sip_pixelxy2radec(wcs, crpix0, crpix1, &nxref, &nyref);

    // RA,DEC coords of new tangent point
	swcs->wcstan.crval[0] = nxref;
	swcs->wcstan.crval[1] = nyref;
	theta = -deg2rad(nxref - crval0); // deltaRA = new minus old RA;
	theta *= sin(deg2rad(nyref));  // multiply by the sin of the NEW Dec; at equator this correctly evals to zero
	sintheta = sin(theta);
	costheta = cos(theta);

	// Restore crpix
	wcs->wcstan.crpix[0] = crpix0;
	wcs->wcstan.crpix[1] = crpix1;

	// Fix the CD matrix since "northwards" has changed due to moving RA
	newCD[0][0] = costheta * swcs->wcstan.cd[0][0] - sintheta * swcs->wcstan.cd[0][1];
	newCD[0][1] = sintheta * swcs->wcstan.cd[0][0] + costheta * swcs->wcstan.cd[0][1];
	newCD[1][0] = costheta * swcs->wcstan.cd[1][0] - sintheta * swcs->wcstan.cd[1][1];
	newCD[1][1] = sintheta * swcs->wcstan.cd[1][0] + costheta * swcs->wcstan.cd[1][1];
	swcs->wcstan.cd[0][0] = newCD[0][0];
	swcs->wcstan.cd[0][1] = newCD[0][1];
	swcs->wcstan.cd[1][0] = newCD[1][0];
	swcs->wcstan.cd[1][1] = newCD[1][1];

	// go into sanity_check and try this:
	// make something one sq. degree with DEC=89degrees and north up
	//make xy, convert to to RA/DEC
	//shift the WCS by .5 degrees of RA (give this in terms of pixels), use this new WCS to convert RA/DEC back to pixels
	//compare those pixels with pixels that have been shifted by .5 degrees worth of pixels to the left (in x direction only)

	return swcs;
}

static sip_t* do_entire_shift_operation(tweak_t* t, double rho) {
	sip_t* swcs;
	get_shift(t->x, t->y, t->n,
	          t->x_ref, t->y_ref, t->n_ref,
	          rho*t->mindx, rho*t->mindy, rho*t->maxdx, rho*t->maxdy,
	          &t->xs, &t->ys);
	swcs = wcs_shift(t->sip, t->xs, t->ys);
	sip_free(t->sip);
	t->sip = swcs;
	return NULL;
}

/* This function is intended only for initializing newly allocated tweak
 * structures, NOT for operating on existing ones.*/
void tweak_init(tweak_t* t) {
	memset(t, 0, sizeof(tweak_t));
    t->sip = sip_create();
}

tweak_t* tweak_new() {
	tweak_t* t = malloc(sizeof(tweak_t));
	tweak_init(t);
	return t;
}

void tweak_iterate_to_order(tweak_t* t, int maxorder, int iterations) {
    int order;
    int k;

    for (order=1; order<=maxorder; order++) {
        logverb("\n");
        logverb("--------------------------------\n");
        logverb("Order %i\n", order);
        logverb("--------------------------------\n");

        t->sip->a_order  = t->sip->b_order  = order;
        //t->sip->ap_order = t->sip->bp_order = order;
        tweak_go_to(t, TWEAK_HAS_CORRESPONDENCES);

        for (k=0; k<iterations; k++) {
            logverb("\n");
            logverb("--------------------------------\n");
            logverb("Iterating tweak: order %i, step %i\n", order, k);
            t->state &= ~TWEAK_HAS_LINEAR_CD;
            tweak_go_to(t, TWEAK_HAS_LINEAR_CD);
            tweak_clear_correspondences(t);
        }
    }
}

#define CHECK_STATE(x) if (state & x) sl_append(s, #x)

static char* state_string(unsigned int state) {
    sl* s = sl_new(4);
    char* str;
	CHECK_STATE(TWEAK_HAS_SIP);
	CHECK_STATE(TWEAK_HAS_IMAGE_XY);
	CHECK_STATE(TWEAK_HAS_IMAGE_XYZ);
	CHECK_STATE(TWEAK_HAS_IMAGE_AD);
	CHECK_STATE(TWEAK_HAS_REF_XY);
	CHECK_STATE(TWEAK_HAS_REF_XYZ);
	CHECK_STATE(TWEAK_HAS_REF_AD);
	CHECK_STATE(TWEAK_HAS_CORRESPONDENCES);
	CHECK_STATE(TWEAK_HAS_COARSLY_SHIFTED);
	CHECK_STATE(TWEAK_HAS_FINELY_SHIFTED);
	CHECK_STATE(TWEAK_HAS_REALLY_FINELY_SHIFTED);
	CHECK_STATE(TWEAK_HAS_LINEAR_CD);
    str = sl_join(s, " ");
    sl_free2(s);
    return str;
}

char* tweak_get_state_string(const tweak_t* t) {
    return state_string(t->state);
}

#undef CHECK_STATE

void tweak_clear_correspondences(tweak_t* t) {
	if (t->state & TWEAK_HAS_CORRESPONDENCES) {
		// our correspondences are also now toast
		assert(t->image);
		assert(t->ref);
		assert(t->dist2);
		il_free(t->image);
		il_free(t->ref);
		dl_free(t->dist2);
		if (t->weight)
			dl_free(t->weight);
		t->image    = NULL;
		t->ref      = NULL;
		t->dist2    = NULL;
		t->weight   = NULL;
		t->state &= ~TWEAK_HAS_CORRESPONDENCES;
	}
	assert(!t->image);
	assert(!t->ref);
	assert(!t->dist2);
	assert(!t->weight);
}

void tweak_clear_on_sip_change(tweak_t* t) {
    // tweak_clear_correspondences(t);
	tweak_clear_image_ad(t);
	tweak_clear_ref_xy(t);
	tweak_clear_image_xyz(t);
}

// ref_xy are the catalog star positions in image coordinates
void tweak_clear_ref_xy(tweak_t* t) {
	if (t->state & TWEAK_HAS_REF_XY) {
		//assert(t->x_ref);
		free(t->x_ref);
		//assert(t->y_ref);
		t->x_ref = NULL;
		free(t->y_ref);
		t->y_ref = NULL;
		t->state &= ~TWEAK_HAS_REF_XY;
    }
    assert(!t->x_ref);
    assert(!t->y_ref);
}

// radec of catalog stars
void tweak_clear_ref_ad(tweak_t* t) {
	if (t->state & TWEAK_HAS_REF_AD) {
		assert(t->a_ref);
		free(t->a_ref);
		t->a_ref = NULL;
		assert(t->d_ref);
		free(t->d_ref);
		t->d_ref = NULL;
		t->n_ref = 0;
		tweak_clear_correspondences(t);
		tweak_clear_ref_xy(t);
		t->state &= ~TWEAK_HAS_REF_AD;
    }
    assert(!t->a_ref);
    assert(!t->d_ref);
}

// image objs in ra,dec according to current tweak
void tweak_clear_image_ad(tweak_t* t) {
	if (t->state & TWEAK_HAS_IMAGE_AD) {
		assert(t->a);
		free(t->a);
		t->a = NULL;
		assert(t->d);
		free(t->d);
		t->d = NULL;
		t->state &= ~TWEAK_HAS_IMAGE_AD;
    }
    assert(!t->a);
    assert(!t->d);
}

void tweak_clear_image_xyz(tweak_t* t) {
	if (t->state & TWEAK_HAS_IMAGE_XYZ) {
		assert(t->xyz);
		free(t->xyz);
		t->xyz = NULL;
		t->state &= ~TWEAK_HAS_IMAGE_XYZ;
    }
    assert(!t->xyz);
}

void tweak_clear_image_xy(tweak_t* t) {
	if (t->state & TWEAK_HAS_IMAGE_XY) {
		assert(t->x);
		free(t->x);
		t->x = NULL;
		assert(t->y);
		free(t->y);
		t->y = NULL;
		t->state &= ~TWEAK_HAS_IMAGE_XY;
	}
    assert(!t->x);
    assert(!t->y);
}

// tell us (from outside tweak) where the catalog stars are
void tweak_push_ref_ad(tweak_t* t, const double* a, const double *d, int n) {
	assert(a);
	assert(d);
	assert(n);
	tweak_clear_ref_ad(t);
	assert(!t->a_ref);
	assert(!t->d_ref);
	t->a_ref = malloc(sizeof(double) * n);
	t->d_ref = malloc(sizeof(double) * n);
	memcpy(t->a_ref, a, n*sizeof(double));
	memcpy(t->d_ref, d, n*sizeof(double));
	t->n_ref = n;
	t->state |= TWEAK_HAS_REF_AD;
}

void tweak_push_ref_ad_array(tweak_t* t, const double* ad, int n) {
    int i;
	assert(ad);
	assert(n);
	tweak_clear_ref_ad(t);
	assert(!t->a_ref);
	assert(!t->d_ref);
	t->a_ref = malloc(sizeof(double) * n);
	t->d_ref = malloc(sizeof(double) * n);
    for (i=0; i<n; i++) {
        t->a_ref[i] = ad[2*i + 0];
        t->d_ref[i] = ad[2*i + 1];
    }
	t->n_ref = n;
	t->state |= TWEAK_HAS_REF_AD;
}

static void ref_xyz_from_ad(tweak_t* t) {
	int i;
	assert(t->state & TWEAK_HAS_REF_AD);
	assert(!t->xyz_ref);
	t->xyz_ref = malloc(sizeof(double) * 3 * t->n_ref);
    assert(t->xyz_ref);
	for (i = 0; i < t->n_ref; i++)
		radecdeg2xyzarr(t->a_ref[i], t->d_ref[i], t->xyz_ref + 3 * i);
	t->state |= TWEAK_HAS_REF_XYZ;
}

static void ref_ad_from_xyz(tweak_t* t) {
	int i, n;
	assert(t->state & TWEAK_HAS_REF_XYZ);
	assert(!t->a_ref);
	assert(!t->d_ref);
    n = t->n_ref;
    t->a_ref = malloc(sizeof(double) * n);
    t->d_ref = malloc(sizeof(double) * n);
    assert(t->a_ref);
    assert(t->d_ref);
	for (i=0; i<n; i++)
        xyzarr2radecdeg(t->xyz_ref + 3*i, t->a_ref + i, t->d_ref + i);
	t->state |= TWEAK_HAS_REF_XYZ;
}

// tell us (from outside tweak) where the catalog stars are
void tweak_push_ref_xyz(tweak_t* t, const double* xyz, int n) {
	assert(xyz);
	assert(n);
	tweak_clear_ref_ad(t);
	assert(!t->xyz_ref);
	t->xyz_ref = malloc(sizeof(double) * 3 * n);
	assert(t->xyz_ref);
	memcpy(t->xyz_ref, xyz, 3*n*sizeof(double));
	t->n_ref = n;
	t->state |= TWEAK_HAS_REF_XYZ;
}

void tweak_push_image_xy(tweak_t* t, const starxy_t* xy) {
    tweak_clear_image_xy(t);
    t->x = starxy_copy_x(xy);
    t->y = starxy_copy_y(xy);
    t->n = starxy_n(xy);
	t->state |= TWEAK_HAS_IMAGE_XY;
}

void tweak_skip_shift(tweak_t* t) {
	t->state |= (TWEAK_HAS_COARSLY_SHIFTED | TWEAK_HAS_FINELY_SHIFTED |
	             TWEAK_HAS_REALLY_FINELY_SHIFTED);
}

// DualTree RangeSearch callback. We want to keep track of correspondences.
// Potentially the matching could be many-to-many; we allow this and hope the
// optimizer can take care of it.
static void dtrs_match_callback(void* extra, int image_ind, int ref_ind, double dist2) {
	tweak_t* t = extra;
	image_ind = kdtree_permute(t->kd_image, image_ind);
	ref_ind   = kdtree_permute(t->kd_ref,   ref_ind);
	il_append(t->image, image_ind);
	il_append(t->ref, ref_ind);
	dl_append(t->dist2, dist2);
	if (t->weight)
		dl_append(t->weight, exp(-dist2 / (2.0 * t->jitterd2)));
}

void tweak_push_correspondence_indices(tweak_t* t, il* image, il* ref, dl* distsq, dl* weight) {
	t->image = image;
	t->ref = ref;
	t->dist2 = distsq;
	t->weight = weight;
	t->state |= TWEAK_HAS_CORRESPONDENCES;
}

// The jitter is in radians
static void find_correspondences(tweak_t* t, double jitter) {
	double dist;
	double* data_image = malloc(sizeof(double) * t->n * 3);
	double* data_ref = malloc(sizeof(double) * t->n_ref * 3);

	assert(t->state & TWEAK_HAS_IMAGE_XYZ);
	assert(t->state & TWEAK_HAS_REF_XYZ);
	tweak_clear_correspondences(t);

	memcpy(data_image, t->xyz, 3*t->n*sizeof(double));
	memcpy(data_ref, t->xyz_ref, 3*t->n_ref*sizeof(double));

	t->kd_image = kdtree_build(NULL, data_image, t->n, 3, 4, KDTT_DOUBLE,
	                           KD_BUILD_BBOX);

	t->kd_ref = kdtree_build(NULL, data_ref, t->n_ref, 3, 4, KDTT_DOUBLE,
	                         KD_BUILD_BBOX);

	// Storage for correspondences
	t->image = il_new(600);
	t->ref = il_new(600);
	t->dist2 = dl_new(600);
	if (t->weighted_fit)
		t->weight = dl_new(600);

	dist = rad2dist(jitter);

	logverb("search radius = %g arcsec\n", rad2arcsec(jitter));

	// Find closest neighbours
	dualtree_rangesearch(t->kd_image, t->kd_ref,
	                     RANGESEARCH_NO_LIMIT, dist, FALSE, NULL,
	                     dtrs_match_callback, t,
	                     NULL, NULL);

	kdtree_free(t->kd_image);
	kdtree_free(t->kd_ref);
	t->kd_image = NULL;
	t->kd_ref = NULL;
	free(data_image);
	free(data_ref);

	logverb("Number of correspondences: %d\n", il_size(t->image));
}

static double correspondences_rms_arcsec(tweak_t* t, int weighted) {
	double err2 = 0.0;
	int i;
	double totalweight = 0.0;
	for (i=0; i<il_size(t->image); i++) {
		double imgxyz[3];
		double refxyz[3];
		double weight;
        int refi, imgi;
		if (weighted && t->weight)
			weight = dl_get(t->weight, i);
        else
			weight = 1.0;
		totalweight += weight;

        imgi = il_get(t->image, i);
		sip_pixelxy2xyzarr(t->sip, t->x[imgi], t->y[imgi], imgxyz);

        refi = il_get(t->ref, i);
		radecdeg2xyzarr(t->a_ref[refi], t->d_ref[refi], refxyz);

		err2 += weight * distsq(imgxyz, refxyz, 3);
	}
	return distsq2arcsec( err2 / totalweight );
}

// in arcseconds^2 on the sky (chi-sq)
static double figure_of_merit(tweak_t* t, double *rmsX, double *rmsY) {
	double sqerr = 0.0;
	int i;
	for (i = 0; i < il_size(t->image); i++) {
		double a, d;
		double xyzpt[3];
		double xyzpt_ref[3];
		sip_pixelxy2radec(t->sip, t->x[il_get(t->image, i)],
		                  t->y[il_get(t->image, i)], &a, &d);

		// xref and yref should be intermediate WC's not image x and y!
		radecdeg2xyzarr(a, d, xyzpt);
		radecdeg2xyzarr(t->a_ref[il_get(t->ref, i)],
		                t->d_ref[il_get(t->ref, i)], xyzpt_ref);
        sqerr += distsq(xyzpt, xyzpt_ref, 3);
	}
	return rad2arcsec(1)*rad2arcsec(1)*sqerr;
}

static double figure_of_merit2(tweak_t* t) {
    // find error in pixels^2
	double sqerr = 0.0;
	int i;
	for (i = 0; i < il_size(t->image); i++) {
		double x, y, dx, dy;
		Unused anbool ok;
		ok = sip_radec2pixelxy(t->sip, t->a_ref[il_get(t->ref, i)], t->d_ref[il_get(t->ref, i)], &x, &y);
		assert(ok);
		dx = t->x[il_get(t->image, i)] - x;
		dy = t->y[il_get(t->image, i)] - y;
		sqerr += dx * dx + dy * dy;
	}
    // convert to arcsec^2
    return sqerr * square(sip_pixel_scale(t->sip));
}

static void invert_sip_polynomial(tweak_t* t) {
	sip_compute_inverse_polynomials(t->sip, 0, 0, 0, 0, 0, 0);
}

/*
    T_0(x) = 1
    T_1(x) = x
    T_2(x) = 2x^2 - 1
    T_3(x) = 4x^3 - 3x
    T_4(x) = 8x^4 - 8x^2 + 1
    T_5(x) = 16x^5 - 20x^3 + 5x
    T_6(x) = 32x^6 - 48x^4 + 18x^2 - 1
    T_7(x) = 64x^7 - 112x^5 + 56x^3 - 7x
    T_8(x) = 128x^8 - 256x^6 + 160x^4 - 32x^2 + 1
    T_9(x) = 256x^9 - 576x^7 + 432x^5 - 120x^3 + 9x
 */
static double tchebyshev(double x, int o) {
	switch (o) {
	case 0: return   1;
	case 1: return   x;
	case 2: return   2.*pow(x, 2) - 1;
	case 3: return   4.*pow(x, 3) -   3*x;
	case 4: return   8.*pow(x, 4) -   8.*pow(x, 2) +   1;
	case 5: return  16.*pow(x, 5) -  20.*pow(x, 3) +   5.*x;
	case 6: return  32.*pow(x, 6) -  48.*pow(x, 4) +  18.*pow(x,2) -  1;
	case 7: return  64.*pow(x, 7) - 112.*pow(x, 5) +  56.*pow(x,3) -  7.*x;
	case 8: return 128.*pow(x, 8) - 256.*pow(x, 6) + 160.*pow(x,4) -  32.*pow(x,2) + 1;
	case 9: return 256.*pow(x, 9) - 576.*pow(x, 7) + 432.*pow(x,5) - 120.*pow(x,3) + 9.*x;
	}
	assert(0);
	return 0.0;
}

//static void tchebyshev_tweak(tweak_t* t, int W, int H) {
void tchebyshev_tweak(tweak_t* t, int W, int H) {
	int Torder;
	double xyzcrval[3];
	double cdinv[2][2];
	double sx, sy, sU, sV, su, sv;
	sip_t* swcs;
	int M, N;
	int i, j, p, q, order;
	double totalweight;
	int rtn;
	gsl_matrix *mA;
	gsl_vector *b1, *b2, *x1, *x2;
	gsl_vector *r1=NULL, *r2=NULL;

	// a_order and b_order should be the same!
	assert(t->sip->a_order == t->sip->b_order);
	Torder = t->sip->a_order;
	// We need at least the linear terms to compute CD.
	if (Torder < 1)
		Torder = 1;

	// The SIP coefficients form an (order x order) upper triangular
	// matrix missing the 0,0 element.
	N = (Torder + 1) * (Torder + 2) / 2;

	// number of correspondences
	M = il_size(t->image);

    if (M < N) {
        logmsg("Too few correspondences for the SIP order specified (%i < %i)\n", M, N);
        return;
    }

	logverb("RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 0));
	logverb("Weighted RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 1));

	mA = gsl_matrix_alloc(M, N);
	b1 = gsl_vector_alloc(M);
	b2 = gsl_vector_alloc(M);
	assert(mA);
	assert(b1);
	assert(b2);

	// Fill in matrix mA:
	radecdeg2xyzarr(t->sip->wcstan.crval[0], t->sip->wcstan.crval[1], xyzcrval);
	totalweight = 0.0;
	for (i=0; i<M; i++) {
        int refi;
        double x=0, y=0;
        double xyzpt[3];
        double weight = 1.0;
        double u;
        double v;
        Unused anbool ok;

		// convert to Tchebyshev domain: [-1,1]
        u = (t->x[il_get(t->image, i)] - t->sip->wcstan.crpix[0]) / (double)W;
		v = (t->y[il_get(t->image, i)] - t->sip->wcstan.crpix[1]) / (double)H;

        if (t->weighted_fit) {
            weight = dl_get(t->weight, i);
            assert(weight >= 0.0);
            assert(weight <= 1.0);
            totalweight += weight;
        }

        j = 0;
        for (order=0; order<=Torder; order++) {
            for (q=0; q<=order; q++) {
                p = order - q;
                assert(j >= 0);
                assert(j < N);
                assert(p >= 0);
                assert(q >= 0);
                assert(p + q <= Torder);
                gsl_matrix_set(mA, i, j, weight * tchebyshev(u, p) * tchebyshev(v, q));
                j++;
            }
        }
        assert(j == N);

        // The shift - aka (0,0) - SIP coefficient must be 1.
        assert(gsl_matrix_get(mA, i, 0) == 1.0 * weight);
        assert(fabs(gsl_matrix_get(mA, i, 1) - u * weight) < 1e-12);
        assert(fabs(gsl_matrix_get(mA, i, 2) - v * weight) < 1e-12);

        // B contains Intermediate World Coordinates (in degrees)
        refi = il_get(t->ref, i);
        radecdeg2xyzarr(t->a_ref[refi], t->d_ref[refi], xyzpt);
		// tangent-plane projection
        ok = star_coords(xyzpt, xyzcrval, TRUE, &x, &y);
        assert(ok);

        gsl_vector_set(b1, i, weight * rad2deg(x));
        gsl_vector_set(b2, i, weight * rad2deg(y));
    }

	if (t->weighted_fit)
		logverb("Total weight: %g\n", totalweight);


	// Solve the equation.
    //rtn = gslutils_solve_leastsquares_v(mA, 2, b1, &x1, &r1, b2, &x2, &r2);
	rtn = gslutils_solve_leastsquares_v(mA, 2, b1, &x1, NULL, b2, &x2, NULL);
	if (rtn) {
        ERROR("Failed to solve tweak inversion matrix equation!");
        return;
    }

	// Row 0 of X are the shift (p=0, q=0) terms.
	// Row 1 of X are the terms that multiply "u".
	// Row 2 of X are the terms that multiply "v".

	// Grab CD.
	t->sip->wcstan.cd[0][0] = gsl_vector_get(x1, 1);
	t->sip->wcstan.cd[1][0] = gsl_vector_get(x2, 1);
	t->sip->wcstan.cd[0][1] = gsl_vector_get(x1, 2);
	t->sip->wcstan.cd[1][1] = gsl_vector_get(x2, 2);

	// Compute inv(CD)
	i = invert_2by2_arr((const double*)(t->sip->wcstan.cd), (double*)cdinv);
	assert(i == 0);

	// Grab the shift.
	sx = gsl_vector_get(x1, 0);
	sy = gsl_vector_get(x2, 0);

	// Extract the SIP coefficients.
	//  (this includes the 0 and 1 order terms, which we later overwrite)
	j = 0;
	for (order=0; order<=Torder; order++) {
		for (q=0; q<=order; q++) {
			p = order - q;
			assert(j >= 0);
			assert(j < N);
			assert(p >= 0);
			assert(q >= 0);
			assert(p + q <= Torder);

			t->sip->a[p][q] =
				cdinv[0][0] * gsl_vector_get(x1, j) +
				cdinv[0][1] * gsl_vector_get(x2, j);

			t->sip->b[p][q] =
				cdinv[1][0] * gsl_vector_get(x1, j) +
				cdinv[1][1] * gsl_vector_get(x2, j);

			j++;
		}
	}
	assert(j == N);

	// We have already dealt with the shift and linear terms, so zero them out
	// in the SIP coefficient matrix.
	t->sip->a_order = Torder;
	t->sip->b_order = Torder;
	t->sip->a[0][0] = 0.0;
	t->sip->b[0][0] = 0.0;
	t->sip->a[0][1] = 0.0;
	t->sip->a[1][0] = 0.0;
	t->sip->b[0][1] = 0.0;
	t->sip->b[1][0] = 0.0;

	printf("Tchebyshev coefficient matrix for X:\n");
	for (i=0; i<Torder; i++) {
		for (j=0; j<Torder; j++) {
			printf("% 8.4f   ", t->sip->a[i][j]);
		}
		printf("\n");
	}
	printf("Tchebyshev coefficient matrix for Y:\n");
	for (i=0; i<Torder; i++) {
		for (j=0; j<Torder; j++) {
			printf("% 8.4f   ", t->sip->b[i][j]);
		}
		printf("\n");
	}



	///////// FIXME -- most things below here don't apply to Tchebshev polys...

	invert_sip_polynomial(t);

	logverb("Applying shift of sx,sy = %g,%g deg to CRVAL and CD.\n", sx, sy);
	sU =
		cdinv[0][0] * sx +
		cdinv[0][1] * sy;
	sV =
		cdinv[1][0] * sx +
		cdinv[1][1] * sy;
	sip_calc_inv_distortion(t->sip, sU, sV, &su, &sv);

	debug("sx = %g, sy = %g\n", sx, sy);
	debug("sU = %g, sV = %g\n", sU, sV);
	debug("su = %g, sv = %g\n", su, sv);

	swcs = wcs_shift(t->sip, -su, -sv);
	memcpy(t->sip, swcs, sizeof(sip_t));
	sip_free(swcs);

	// recalc using new SIP
    tweak_clear_on_sip_change(t);
	tweak_go_to(t, TWEAK_HAS_IMAGE_AD);
	tweak_go_to(t, TWEAK_HAS_REF_XY);

	logverb("RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 0));
	logverb("Weighted RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 1));

	if (r1)
		gsl_vector_free(r1);
	if (r2)
		gsl_vector_free(r2);

	gsl_matrix_free(mA);
	gsl_vector_free(b1);
	gsl_vector_free(b2);
	gsl_vector_free(x1);
	gsl_vector_free(x2);
}



// FIXME: adapt this function to take as input the correspondences to use VVVVV
//    wic is World Intermediate Coordinates, either along ra or dec
//       i.e. canonical image coordinates
//    wic_corr is the list of corresponding indexes for wip
//    pix_corr is the list of corresponding indexes for pixels
//    pix is raw image pixels (either u or v)
//    siporder is the sip order up to MAXORDER (defined in sip.h)
//    the correspondences are passed so that we can stick RANSAC around the whole
//    thing for better estimation.

// Run a polynomial tweak
static void do_sip_tweak(tweak_t* t) {
	int sip_order, sip_coeffs;
	double xyzcrval[3];
	double cdinv[2][2];
	double sx, sy, sU, sV, su, sv;
	sip_t* swcs;
	int M, N;
	int i, j, p, q, order;
	double totalweight;
	int rtn;
	gsl_matrix *mA;
	gsl_vector *b1, *b2, *x1, *x2;
	gsl_vector *r1=NULL, *r2=NULL;

	// a_order and b_order should be the same!
	assert(t->sip->a_order == t->sip->b_order);
	sip_order = t->sip->a_order;

	// We need at least the linear terms to compute CD.
	if (sip_order < 1)
		sip_order = 1;

	// The SIP coefficients form an (order x order) upper triangular
	// matrix missing the 0,0 element.
	sip_coeffs = (sip_order + 1) * (sip_order + 2) / 2;

	M = il_size(t->image);
	N = sip_coeffs;

    if (M < N) {
        logmsg("Too few correspondences for the SIP order specified (%i < %i)\n", M, N);
        return;
    }

	mA = gsl_matrix_alloc(M, N);
	b1 = gsl_vector_alloc(M);
	b2 = gsl_vector_alloc(M);
	assert(mA);
	assert(b1);
	assert(b2);

	debug("do_sip_tweak starting.\n");
	logverb("RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 0));
	logverb("Weighted RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 1));

	/*
     *  We use a clever trick to estimate CD, A, and B terms in two
     *  seperated least squares fits, then finding A and B by multiplying
     *  the found parameters by CD inverse.
     * 
     *  Rearranging the SIP equations (see sip.h) we get the following
     *  matrix operation to compute x and y in world intermediate
     *  coordinates, which is convienently written in a way which allows
     *  least squares estimation of CD and terms related to A and B.
     * 
     *  First use the x's to find the first set of parametetrs
     * 
     *     +--------------------- Intermediate world coordinates in DEGREES
     *     |          +--------- Pixel coordinates u and v in PIXELS
     *     |          |     +--- Polynomial u,v terms in powers of PIXELS
     *     v          v     v
     *   ( x1 )   ( 1 u1 v1 p1 )   (sx              )
     *   ( x2 ) = ( 1 u2 v2 p2 ) * (cd11            ) : cd11 is a scalar, degrees per pixel
     *   ( x3 )   ( 1 u3 v3 p3 )   (cd12            ) : cd12 is a scalar, degrees per pixel
     *   ( ...)   (   ...    )     (cd11*A + cd12*B ) : cd11*A and cs12*B are mixture of SIP terms (A,B) and CD matrix (cd11,cd12)
     * 
     *  Then find cd21 and cd22 with the y's
     * 
     *   ( y1 )   ( 1 u1 v1 p1 )   (sy              )
     *   ( y2 ) = ( 1 u2 v2 p2 ) * (cd21            ) : scalar, degrees per pixel
     *   ( y3 )   ( 1 u3 v3 p3 )   (cd22            ) : scalar, degrees per pixel
     *   ( ...)   (   ...    )     (cd21*A + cd22*B ) : mixture of SIP terms (A,B) and CD matrix (cd21,cd22)
     * 
     *  These are both standard least squares problems which we solve with
     *  QR decomposition, ie
     *      min_{cd,A,B} || x - [1,u,v,p]*[s;cd;cdA+cdB]||^2 with
     *  x reference, cd,A,B unrolled parameters.
     * 
     *  We get back (for x) a vector of optimal
     *    [sx;cd11;cd12; cd11*A + cd12*B]
     *  Now we can pull out sx, cd11 and cd12 from the beginning of this vector,
     *  and call the rest of the vector [cd11*A] + [cd12*B];
     *  similarly for the y fit, we get back a vector of optimal
     *    [sy;cd21;cd22; cd21*A + cd22*B]
     *  once we have all those we can figure out A and B as follows
     *                   -1
     *    A' = [cd11 cd12]    *  [cd11*A' + cd12*B']
     *    B'   [cd21 cd22]       [cd21*A' + cd22*B']
     * 
     *  which recovers the A and B's.
     *
     */

	/*
     *  Dustin's interpretation of the above:
     *  We want to solve:
     * 
     *     min || b[M-by-1] - A[M-by-N] x[N-by-1] ||_2
     * 
     *  M = the number of correspondences.
     *  N = the number of SIP terms.
     *
     * And we want an overdetermined system, so M >= N.
     * 
     *           [ 1  u_1   v_1  u_1^2  u_1 v_1  v_1^2  ... ]
     *    mA  =  [ 1  u_2   v_2  u_2^2  u_2 v_2  v_2^2  ... ]
     *           [           ......                         ]
	 *
	 * Where (u_i, v_i) are *undistorted* pixel positions minus CRPIX.
	 *
     *  The answers we want are:
     *
     *         [ sx                  ]
     *    x1 = [ cd11                ]
     *         [ cd12                ]
	 *         [      (A)        (B) ]
     *         [ cd11*(A) + cd12*(B) ]
	 *         [      (A)        (B) ]
     *
     *         [ sy                  ]
     *    x2 = [ cd21                ]
     *         [ cd22                ]
	 *         [      (A)        (B) ]
     *         [ cd21*(A) + cd22*(B) ]
	 *         [      (A)        (B) ]
	 *
	 * And the target vectors are the intermediate world coords of the
	 * reference stars, in degrees.
     *
     *         [ ix_1 ]
     *    b1 = [ ix_2 ]
     *         [ ...  ]
     *
     *         [ iy_1 ]
     *    b2 = [ iy_2 ]
     *         [ ...  ]
     *
     *
     *  (where A and B are tall vectors of SIP coefficients of order 2
     *  and above)
     *
     */

	// Fill in matrix mA:
	radecdeg2xyzarr(t->sip->wcstan.crval[0], t->sip->wcstan.crval[1], xyzcrval);
	totalweight = 0.0;
	for (i=0; i<M; i++) {
        int refi;
        double x=0, y=0;
        double xyzpt[3];
        double weight = 1.0;
        double u;
        double v;
        Unused anbool ok;

        u = t->x[il_get(t->image, i)] - t->sip->wcstan.crpix[0];
        v = t->y[il_get(t->image, i)] - t->sip->wcstan.crpix[1];

        if (t->weighted_fit) {
            weight = dl_get(t->weight, i);
            assert(weight >= 0.0);
            assert(weight <= 1.0);
            totalweight += weight;
        }

        /* The coefficients are stored in this order:
         *   p q
         *  (0,0) = 1     <- order 0
         *  (1,0) = u     <- order 1
         *  (0,1) = v
         *  (2,0) = u^2   <- order 2
         *  (1,1) = uv
         *  (0,2) = v^2
         *  ...
         */

        j = 0;
        for (order=0; order<=sip_order; order++) {
            for (q=0; q<=order; q++) {
                p = order - q;
                assert(j >= 0);
                assert(j < N);
                assert(p >= 0);
                assert(q >= 0);
                assert(p + q <= sip_order);
                gsl_matrix_set(mA, i, j, weight * pow(u, (double)p) * pow(v, (double)q));
                j++;
            }
        }
        assert(j == N);

        // The shift - aka (0,0) - SIP coefficient must be 1.
        assert(gsl_matrix_get(mA, i, 0) == 1.0 * weight);
        assert(fabs(gsl_matrix_get(mA, i, 1) - u * weight) < 1e-12);
        assert(fabs(gsl_matrix_get(mA, i, 2) - v * weight) < 1e-12);

        // B contains Intermediate World Coordinates (in degrees)
        refi = il_get(t->ref, i);
        radecdeg2xyzarr(t->a_ref[refi], t->d_ref[refi], xyzpt);
        ok = star_coords(xyzpt, xyzcrval, TRUE, &x, &y);
		// tangent-plane projection
        assert(ok);

        gsl_vector_set(b1, i, weight * rad2deg(x));
        gsl_vector_set(b2, i, weight * rad2deg(y));
    }

	if (t->weighted_fit)
		logverb("Total weight: %g\n", totalweight);


	// Solve the equation.
    //rtn = gslutils_solve_leastsquares_v(mA, 2, b1, &x1, &r1, b2, &x2, &r2);
	rtn = gslutils_solve_leastsquares_v(mA, 2, b1, &x1, NULL, b2, &x2, NULL);
	if (rtn) {
        ERROR("Failed to solve tweak inversion matrix equation!");
        return;
    }

	// Row 0 of X are the shift (p=0, q=0) terms.
	// Row 1 of X are the terms that multiply "u".
	// Row 2 of X are the terms that multiply "v".

	// Grab CD.
	t->sip->wcstan.cd[0][0] = gsl_vector_get(x1, 1);
	t->sip->wcstan.cd[1][0] = gsl_vector_get(x2, 1);
	t->sip->wcstan.cd[0][1] = gsl_vector_get(x1, 2);
	t->sip->wcstan.cd[1][1] = gsl_vector_get(x2, 2);

	// Compute inv(CD)
	i = invert_2by2_arr((const double*)(t->sip->wcstan.cd), (double*)cdinv);
	assert(i == 0);

	// Grab the shift.
	sx = gsl_vector_get(x1, 0);
	sy = gsl_vector_get(x2, 0);

	// Extract the SIP coefficients.
	//  (this includes the 0 and 1 order terms, which we later overwrite)
	j = 0;
	for (order=0; order<=sip_order; order++) {
		for (q=0; q<=order; q++) {
			p = order - q;
			assert(j >= 0);
			assert(j < N);
			assert(p >= 0);
			assert(q >= 0);
			assert(p + q <= sip_order);

			t->sip->a[p][q] =
				cdinv[0][0] * gsl_vector_get(x1, j) +
				cdinv[0][1] * gsl_vector_get(x2, j);

			t->sip->b[p][q] =
				cdinv[1][0] * gsl_vector_get(x1, j) +
				cdinv[1][1] * gsl_vector_get(x2, j);

			j++;
		}
	}
	assert(j == N);

	// We have already dealt with the shift and linear terms, so zero them out
	// in the SIP coefficient matrix.
	t->sip->a_order = sip_order;
	t->sip->b_order = sip_order;
	t->sip->a[0][0] = 0.0;
	t->sip->b[0][0] = 0.0;
	t->sip->a[0][1] = 0.0;
	t->sip->a[1][0] = 0.0;
	t->sip->b[0][1] = 0.0;
	t->sip->b[1][0] = 0.0;

	invert_sip_polynomial(t);

	// DEBUG
	/*
	printf("\nBEFORE WCS_SHIFT:\n");
	{
		radecdeg2xyzarr(t->sip->wcstan.crval[0], t->sip->wcstan.crval[1], xyzcrval);
		for (i=0; i<M; i++) {
			int refi;
			double x=0, y=0;
			double xyzpt[3];
			double bx, by;
			double axx, axy;
			anbool ok;
			double weight = 1.0;
			refi = il_get(t->ref, i);
			if (t->weighted_fit)
				weight = dl_get(t->weight, i);
			radecdeg2xyzarr(t->a_ref[refi], t->d_ref[refi], xyzpt);
			ok = star_coords(xyzpt, xyzcrval, &y, &x); // tangent-plane projection
			assert(ok);
			bx = rad2deg(x);
			by = rad2deg(y);
			x = t->x[il_get(t->image, i)];
			y = t->y[il_get(t->image, i)];
			sip_pixelxy2iwc(t->sip, x, y, &axx, &axy);
			debug("Resid %i: GSL says (%g, %g), dstn says (%g, %g), weighted (%g, %g)\n",
				  i, gsl_vector_get(r1, i), gsl_vector_get(r2, i),
				  bx - axx, by - axy, weight*(bx-axx), weight*(by-axy));
		}
	}
	 */

	/*
	 if (t->push_crval) {
	 logverb("push_crval. sx,sy = %g,%g\n", sx, sy);
	 assert(0);
	 } else {
	 */
	{
		sU =
			cdinv[0][0] * sx +
			cdinv[0][1] * sy;
		sV =
			cdinv[1][0] * sx +
			cdinv[1][1] * sy;
		logverb("Applying shift of sx,sy = %g,%g deg (%g,%g pix) to CRVAL and CD.\n", sx, sy, sU, sV);
		sip_calc_inv_distortion(t->sip, sU, sV, &su, &sv);

		debug("sx = %g, sy = %g\n", sx, sy);
		debug("sU = %g, sV = %g\n", sU, sV);
		debug("su = %g, sv = %g\n", su, sv);

		swcs = wcs_shift(t->sip, -su, -sv);
		memcpy(t->sip, swcs, sizeof(sip_t));
		sip_free(swcs);
	}

	// recalc using new SIP
    tweak_clear_on_sip_change(t);
	tweak_go_to(t, TWEAK_HAS_IMAGE_AD);
	tweak_go_to(t, TWEAK_HAS_REF_XY);

	logverb("RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 0));
	logverb("Weighted RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 1));

	// DEBUG
	/*
	{
		// compute the residuals via WCS functions on the new SIP

		// resids are:    b[M-by-1] - A[M-by-N] x[N-by-1]

		// B contains Intermediate World Coordinates (in degrees)
		// of ref stars.

		// A x  ~=  sip_pixelxy2iwc( image stars )

		printf("\nAFTER WCS_SHIFT:\n");
		radecdeg2xyzarr(t->sip->wcstan.crval[0], t->sip->wcstan.crval[1], xyzcrval);
		for (i=0; i<M; i++) {
			int refi;
			double x=0, y=0;
			double xyzpt[3];
			double bx, by;
			double axx, axy;
			anbool ok;
			double weight = 1.0;

			refi = il_get(t->ref, i);
			if (t->weighted_fit)
				weight = dl_get(t->weight, i);
			radecdeg2xyzarr(t->a_ref[refi], t->d_ref[refi], xyzpt);
			ok = star_coords(xyzpt, xyzcrval, &y, &x); // tangent-plane projection
			assert(ok);
			bx = rad2deg(x);
			by = rad2deg(y);

			x = t->x[il_get(t->image, i)];
			y = t->y[il_get(t->image, i)];
			sip_pixelxy2iwc(t->sip, x, y, &axx, &axy);

			printf("Resid %i: GSL says (%g, %g), dstn says (%g, %g), weighted (%g, %g)\n",
				   i, gsl_vector_get(r1, i), gsl_vector_get(r2, i),
				   bx - axx, by - axy, weight*(bx-axx), weight*(by-axy));
		}
	}
	 */

	if (r1)
		gsl_vector_free(r1);
	if (r2)
		gsl_vector_free(r2);

	gsl_matrix_free(mA);
	gsl_vector_free(b1);
	gsl_vector_free(b2);
	gsl_vector_free(x1);
	gsl_vector_free(x2);
}

// Really what we want is some sort of fancy dependency system... DTDS!
// Duct-tape dependencey system (DTDS)
#define done(x) t->state |= x; return x;
#define want(x) \
	if (flag == x && t->state & x) \
		return x; \
	else if (flag == x)
#define ensure(x) \
	if (!(t->state & x)) { \
		return tweak_advance_to(t, x); \
	}

unsigned int tweak_advance_to(tweak_t* t, unsigned int flag) {
	want(TWEAK_HAS_IMAGE_AD) {
		int jj;
		ensure(TWEAK_HAS_SIP);
		ensure(TWEAK_HAS_IMAGE_XY);
		debug("Satisfying TWEAK_HAS_IMAGE_AD\n");
		// Convert to ra dec
		assert(!t->a);
		assert(!t->d);
		t->a = malloc(sizeof(double) * t->n);
		t->d = malloc(sizeof(double) * t->n);
		for (jj = 0; jj < t->n; jj++)
			sip_pixelxy2radec(t->sip, t->x[jj], t->y[jj], t->a + jj, t->d + jj);
		done(TWEAK_HAS_IMAGE_AD);
	}

	want(TWEAK_HAS_REF_AD) {
        if (!(t->a_ref && t->d_ref)) {
            ensure(TWEAK_HAS_REF_XYZ);
            debug("Satisfying TWEAK_HAS_REF_AD\n");
            ref_ad_from_xyz(t);
        }
        assert(t->a_ref && t->d_ref);
		done(TWEAK_HAS_REF_AD);
	}

	want(TWEAK_HAS_REF_XYZ) {
        if (!t->xyz_ref) {
            ensure(TWEAK_HAS_REF_AD);
            debug("Satisfying TWEAK_HAS_REF_XYZ\n");
            ref_xyz_from_ad(t);
        }
        assert(t->xyz_ref);
		done(TWEAK_HAS_REF_XYZ);
	}

	want(TWEAK_HAS_REF_XY) {
		int jj;
		ensure(TWEAK_HAS_REF_AD);
		debug("Satisfying TWEAK_HAS_REF_XY\n");
		assert(t->state & TWEAK_HAS_REF_AD);
		assert(t->n_ref);
		assert(!t->x_ref);
		assert(!t->y_ref);
		t->x_ref = malloc(sizeof(double) * t->n_ref);
		t->y_ref = malloc(sizeof(double) * t->n_ref);
		for (jj = 0; jj < t->n_ref; jj++) {
			Unused anbool ok;
			ok = sip_radec2pixelxy(t->sip, t->a_ref[jj], t->d_ref[jj],
								   t->x_ref + jj, t->y_ref + jj);
			assert(ok);
		}
		done(TWEAK_HAS_REF_XY);
	}

	want(TWEAK_HAS_IMAGE_XYZ) {
		int i;
		ensure(TWEAK_HAS_IMAGE_AD);
		debug("Satisfying TWEAK_HAS_IMAGE_XYZ\n");
		assert(!t->xyz);
		t->xyz = malloc(3 * t->n * sizeof(double));
		for (i = 0; i < t->n; i++)
			radecdeg2xyzarr(t->a[i], t->d[i], t->xyz + 3*i);
		done(TWEAK_HAS_IMAGE_XYZ);
	}

	want(TWEAK_HAS_COARSLY_SHIFTED) {
		ensure(TWEAK_HAS_REF_XY);
		ensure(TWEAK_HAS_IMAGE_XY);
		debug("Satisfying TWEAK_HAS_COARSLY_SHIFTED\n");
		get_dydx_range(t->x, t->y, t->n, t->x_ref, t->y_ref, t->n_ref,
		               &t->mindx, &t->mindy, &t->maxdx, &t->maxdy);
		do_entire_shift_operation(t, 1.0);
		tweak_clear_image_ad(t);
		tweak_clear_ref_xy(t);
		done(TWEAK_HAS_COARSLY_SHIFTED);
	}

	want(TWEAK_HAS_FINELY_SHIFTED) {
		ensure(TWEAK_HAS_REF_XY);
		ensure(TWEAK_HAS_IMAGE_XY);
		ensure(TWEAK_HAS_COARSLY_SHIFTED);
		debug("Satisfying TWEAK_HAS_FINELY_SHIFTED\n");
		// Shrink size of hough box
		do_entire_shift_operation(t, 0.3);
		tweak_clear_image_ad(t);
		tweak_clear_ref_xy(t);
		done(TWEAK_HAS_FINELY_SHIFTED);
	}

	want(TWEAK_HAS_REALLY_FINELY_SHIFTED) {
		ensure(TWEAK_HAS_REF_XY);
		ensure(TWEAK_HAS_IMAGE_XY);
		ensure(TWEAK_HAS_FINELY_SHIFTED);
		debug("Satisfying TWEAK_HAS_REALLY_FINELY_SHIFTED\n");
		// Shrink size of hough box
		do_entire_shift_operation(t, 0.03);
		tweak_clear_image_ad(t);
		tweak_clear_ref_xy(t);
		done(TWEAK_HAS_REALLY_FINELY_SHIFTED);
	}

	want(TWEAK_HAS_CORRESPONDENCES) {
		ensure(TWEAK_HAS_REF_XYZ);
		ensure(TWEAK_HAS_IMAGE_XYZ);
		debug("Satisfying TWEAK_HAS_CORRESPONDENCES\n");
		t->jitterd2 = arcsec2distsq(t->jitter);
		find_correspondences(t, 6.0 * arcsec2rad(t->jitter));
		done(TWEAK_HAS_CORRESPONDENCES);
	}

	want(TWEAK_HAS_LINEAR_CD) {
		ensure(TWEAK_HAS_SIP);
		ensure(TWEAK_HAS_REALLY_FINELY_SHIFTED);
		ensure(TWEAK_HAS_REF_XY);
		ensure(TWEAK_HAS_REF_AD);
		ensure(TWEAK_HAS_IMAGE_XY);
		ensure(TWEAK_HAS_CORRESPONDENCES);
		debug("Satisfying TWEAK_HAS_LINEAR_CD\n");
		do_sip_tweak(t);
		tweak_clear_on_sip_change(t);
		done(TWEAK_HAS_LINEAR_CD);
	}

    // small memleak -- but it's a major bug if this happens, so suck it up.
    logerr("die for dependence: %s\n", state_string(flag));
	assert(0);
	return -1;
}

void tweak_push_wcs_tan(tweak_t* t, const tan_t* wcs) {
	memcpy(&(t->sip->wcstan), wcs, sizeof(tan_t));
	t->state |= TWEAK_HAS_SIP;
}

void tweak_go_to(tweak_t* t, unsigned int dest_state) {
	while (!(t->state & dest_state))
		tweak_advance_to(t, dest_state);
}

#define SAFE_FREE(xx) {free((xx)); xx = NULL;}
void tweak_clear(tweak_t* t) {
	if (!t)
		return ;
	SAFE_FREE(t->a);
	SAFE_FREE(t->d);
	SAFE_FREE(t->x);
	SAFE_FREE(t->y);
	SAFE_FREE(t->xyz);
	SAFE_FREE(t->a_ref);
	SAFE_FREE(t->d_ref);
	SAFE_FREE(t->x_ref);
	SAFE_FREE(t->y_ref);
	SAFE_FREE(t->xyz_ref);
	if (t->sip) {
		sip_free(t->sip);
		t->sip = NULL;
	}
	il_free(t->image);
	il_free(t->ref);
	dl_free(t->dist2);
	if (t->weight)
		dl_free(t->weight);
	t->image = NULL;
	t->ref = NULL;
	t->dist2 = NULL;
	t->weight = NULL;
	kdtree_free(t->kd_image);
	kdtree_free(t->kd_ref);
}

void tweak_free(tweak_t* t) {
	tweak_clear(t);
	free(t);
}
