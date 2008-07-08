/*
 This file is part of the Astrometry.net suite.
 Copyright 2006, 2007 Keir Mierle, David W. Hogg, Sam Roweis and Dustin Lang.

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

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>

#include "tweak.h"
#include "healpix.h"
#include "dualtree_rangesearch.h"
#include "kdtree_fits_io.h"
#include "mathutil.h"
#include "log.h"
#include "permutedsort.h"

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

void get_dydx_range(double* ximg, double* yimg, int nimg,
                    double* xcat, double* ycat, int ncat,
                    double *mindx, double *mindy, double *maxdx, double *maxdy)
{
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

void get_shift(double* ximg, double* yimg, int nimg,
               double* xcat, double* ycat, int ncat,
               double mindx, double mindy, double maxdx, double maxdy,
               double* xshift, double* yshift)
{

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
	double crpix0, crpix1, crval0, crval1;
	double nxref, nyref, theta, sintheta, costheta;
	double newCD[2][2]; //the new CD matrix
	sip_t* swcs = malloc(sizeof(sip_t));
	memcpy(swcs, wcs, sizeof(sip_t));

	// Save old vals
	crpix0 = wcs->wcstan.crpix[0];
	crpix1 = wcs->wcstan.crpix[1];
	crval0 = wcs->wcstan.crval[0];
	crval1 = wcs->wcstan.crval[1];

    // compute the desired projection of the new tangent point by
    // shifting the projection of the current tangent point
	wcs->wcstan.crpix[0] += xs;
	wcs->wcstan.crpix[1] += ys;

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

sip_t* do_entire_shift_operation(tweak_t* t, double rho) {
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
}

tweak_t* tweak_new() {
	tweak_t* t = malloc(sizeof(tweak_t));
	tweak_init(t);
	return t;
}

void tweak_print_the_state(unsigned int state) {
	if (state & TWEAK_HAS_SIP )
		printf("TWEAK_HAS_SIP, ");
	if (state & TWEAK_HAS_IMAGE_XY )
		printf("TWEAK_HAS_IMAGE_XY, ");
	if (state & TWEAK_HAS_IMAGE_XYZ )
		printf("TWEAK_HAS_IMAGE_XYZ, ");
	if (state & TWEAK_HAS_IMAGE_AD )
		printf("TWEAK_HAS_IMAGE_AD, ");
	if (state & TWEAK_HAS_REF_XY )
		printf("TWEAK_HAS_REF_XY, ");
	if (state & TWEAK_HAS_REF_XYZ )
		printf("TWEAK_HAS_REF_XYZ, ");
	if (state & TWEAK_HAS_REF_AD )
		printf("TWEAK_HAS_REF_AD, ");
	if (state & TWEAK_HAS_AD_BAR_AND_R )
		printf("TWEAK_HAS_AD_BAR_AND_R, ");
	if (state & TWEAK_HAS_CORRESPONDENCES)
		printf("TWEAK_HAS_CORRESPONDENCES, ");
	if (state & TWEAK_HAS_RUN_OPT )
		printf("TWEAK_HAS_RUN_OPT, ");
	if (state & TWEAK_HAS_RUN_RANSAC_OPT )
		printf("TWEAK_HAS_RUN_RANSAC_OPT, ");
	if (state & TWEAK_HAS_COARSLY_SHIFTED)
		printf("TWEAK_HAS_COARSLY_SHIFTED, ");
	if (state & TWEAK_HAS_FINELY_SHIFTED )
		printf("TWEAK_HAS_FINELY_SHIFTED, ");
	if (state & TWEAK_HAS_REALLY_FINELY_SHIFTED )
		printf("TWEAK_HAS_REALLY_FINELY_SHIFTED, ");
	if (state & TWEAK_HAS_LINEAR_CD )
		printf("TWEAK_HAS_LINEAR_CD, ");
}

void tweak_print_state(tweak_t* t) {
	tweak_print_the_state(t->state);
}

void get_center_and_radius(double* ra, double* dec, int n,
                           double* ra_mean, double* dec_mean, double* radius) {
	double* xyz = malloc(3 * n * sizeof(double));
	double xyz_mean[3] = {0, 0, 0};
	double maxdist2 = 0;
	int i, j;

	for (i = 0; i < n; i++)
		radecdeg2xyzarr(ra[i], dec[i], xyz + 3*i);

	for (i = 0; i < n; i++)  // dumb average
		for (j = 0; j < 3; j++)
			xyz_mean[j] += xyz[3 * i + j];

	normalize_3(xyz_mean);

	for (i = 0; i < n; i++) { // find largest distance from average
		double dist2 = distsq(xyz_mean, xyz + 3*i, 3);
        maxdist2 = MAX(maxdist2, dist2);
	}
	*radius = sqrt(maxdist2);
	xyzarr2radecdeg(xyz_mean, ra_mean, dec_mean);
	free(xyz);
}

void tweak_clear_correspondences(tweak_t* t) {
	if (t->state & TWEAK_HAS_CORRESPONDENCES) {
		// our correspondences are also now toast
		assert(t->image);
		assert(t->ref);
		assert(t->dist2);
		assert(t->included);
		il_free(t->image);
		il_free(t->ref);
		dl_free(t->dist2);
		il_free(t->included);
		if (t->weight)
			dl_free(t->weight);
		t->image    = NULL;
		t->ref      = NULL;
		t->dist2    = NULL;
		t->included = NULL;
		t->weight   = NULL;
		t->state &= ~TWEAK_HAS_CORRESPONDENCES;
	}
	assert(!t->image);
	assert(!t->ref);
	assert(!t->dist2);
	assert(!t->weight);
	assert(!t->included);
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
		assert(t->x_ref);
		free(t->x_ref);
		assert(t->y_ref);
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
void tweak_push_ref_ad(tweak_t* t, double* a, double *d, int n) {
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

// tell us (from outside tweak) where the catalog stars are
void tweak_ref_find_xyz_from_ad(tweak_t* t) {
	int i;
	assert(t->state & TWEAK_HAS_REF_AD);
	assert(!t->xyz_ref);
	t->xyz_ref = malloc(sizeof(double) * 3 * t->n_ref);
	for (i = 0; i < t->n_ref; i++) {
		double *pt = t->xyz_ref + 3 * i;
		radecdeg2xyzarr(t->a_ref[i], t->d_ref[i], pt);
	}
	t->state |= TWEAK_HAS_REF_XYZ;
}

// tell us (from outside tweak) where the catalog stars are
void tweak_push_ref_xyz(tweak_t* t, double* xyz, int n) {
	double *ra, *dec;
	int i;

	tweak_clear_ref_ad(t);

	assert(xyz);
	assert(n);

	assert(!t->xyz_ref);
	t->xyz_ref = malloc(sizeof(double) * 3 * n);
	memcpy(t->xyz_ref, xyz, 3*n*sizeof(double));

	ra = malloc(sizeof(double) * n);
	dec = malloc(sizeof(double) * n);
	assert(ra);
	assert(dec);

	for (i = 0; i < n; i++) {
		double *pt = xyz + 3 * i;
		xyzarr2radecdeg(pt, ra+i, dec+i);
	}

	t->a_ref = ra;
	t->d_ref = dec;
	t->n_ref = n;

	t->state |= TWEAK_HAS_REF_AD;
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

void tweak_print_rms_curve(tweak_t* t) {
    // sort by t->dist2; t->image and t->ref are the correspondences.
    int* perm;
    double* dists;
    int i, N;
    double ssumpix, ssumarcsec;
    dl* allvals;

    tweak_go_to(t, TWEAK_HAS_REF_XY);
    tweak_go_to(t, TWEAK_HAS_IMAGE_XY);
    tweak_go_to(t, TWEAK_HAS_CORRESPONDENCES);

    N = dl_size(t->dist2);
    dists = malloc(N * sizeof(double));
    dl_copy(t->dist2, 0, N, dists);
    perm = permuted_sort(dists, sizeof(double), compare_doubles, NULL, N);
    allvals = dl_new(256);

    ssumpix = ssumarcsec = 0.0;
    for (i=0; i<N; i++) {
        double ix, iy, fx, fy;
        int ind = perm[i];
        int ii, fi;
        double pix2, arcsec2;
        double rmspix, rmsarcsec;

        // index or reference:
        ii = il_get(t->ref, ind);
        ix = t->x_ref[ii];
        iy = t->y_ref[ii];

        // field or image:
        fi = il_get(t->image, ind);
        fx = t->x[fi];
        fy = t->y[fi];

        pix2 = square(fx - ix) + square(fy - iy);
        ssumpix += pix2;
        rmspix = sqrt(ssumpix / (double)(i + 1));
        arcsec2 = square(distsq2arcsec(dl_get(t->dist2, ind)));
        ssumarcsec += arcsec2;
        rmsarcsec = sqrt(ssumarcsec / (double)(i + 1));

        dl_append(allvals, sqrt(pix2));
        dl_append(allvals, sqrt(arcsec2));
        dl_append(allvals, rmsarcsec);
        dl_append(allvals, rmspix);
    }
    /*
     fprintf(stderr, "pixerrs = [");
     for (i=0; i<N; i++)
     fprintf(stderr, "%g, ", dl_get(allvals, i*4 + 0));
     fprintf(stderr, "];\n");

     fprintf(stderr, "arcsecerrs = [");
     for (i=0; i<N; i++)
     fprintf(stderr, "%g, ", dl_get(allvals, i*4 + 1));
     fprintf(stderr, "];\n");

     fprintf(stderr, "pixrms = [");
     for (i=0; i<N; i++)
     fprintf(stderr, "%g, ", dl_get(allvals, i*4 + 2));
     fprintf(stderr, "];\n");

     fprintf(stderr, "arcsecrms = [");
     for (i=0; i<N; i++)
     fprintf(stderr, "%g, ", dl_get(allvals, i*4 + 3));
     fprintf(stderr, "];\n");
     */

    dl_free(allvals);

    free(dists);
    free(perm);
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
	il_append(t->included, 1);

	if (t->weight)
		dl_append(t->weight, exp(-dist2 / (2.0 * t->jitterd2)));
}

// The jitter is in radians
void find_correspondences(tweak_t* t, double jitter) {
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
	t->included = il_new(600);

	dist = rad2dist(jitter);

	logverb("search radius = %g arcsec / %g arcmin / %g deg\n",
	        rad2arcsec(jitter), rad2arcmin(jitter), rad2deg(jitter));
	logverb("distance on the unit sphere: %g\n", dist);

	// Find closest neighbours
	dualtree_rangesearch(t->kd_image, t->kd_ref,
	                     RANGESEARCH_NO_LIMIT, dist, NULL,
	                     dtrs_match_callback, t,
	                     NULL, NULL);

	kdtree_free(t->kd_image);
	kdtree_free(t->kd_ref);
	t->kd_image = NULL;
	t->kd_ref = NULL;
	free(data_image);
	free(data_ref);

	logverb("Number of correspondences: %d\n", dl_size(t->dist2));
}

// in arcseconds^2 on the sky (chi-sq)
double figure_of_merit(tweak_t* t, double *rmsX, double *rmsY) {
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

		if (il_get(t->included, i)) 
            sqerr += distsq(xyzpt, xyzpt_ref, 3);
	}
	return rad2arcsec(1)*rad2arcsec(1)*sqerr;
}

double correspondences_rms_arcsec(tweak_t* t, int weighted) {
	double err2 = 0.0;
	int i;
	double totalweight = 0.0;
	for (i=0; i<il_size(t->image); i++) {
		double imgxyz[3];
		double refxyz[3];
		double weight;
        int refi, imgi;
		if (!il_get(t->included, i))
			continue;
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

double figure_of_merit2(tweak_t* t) {
    // find error in pixels^2
	double sqerr = 0.0;
	int i;
	for (i = 0; i < il_size(t->image); i++) {
		double x, y, dx, dy;
		bool ok;
		ok = sip_radec2pixelxy(t->sip, t->a_ref[il_get(t->ref, i)], t->d_ref[il_get(t->ref, i)], &x, &y);
		assert(ok);
		dx = t->x[il_get(t->image, i)] - x;
		dy = t->y[il_get(t->image, i)] - y;
		sqerr += dx * dx + dy * dy;
	}
    // convert to arcsec^2
    return sqerr * square(sip_pixel_scale(t->sip));
}

// I apologize for the rampant copying and pasting of the polynomial calcs...
void invert_sip_polynomial(tweak_t* t) {
	/*
     basic idea: lay down a grid in image, for each gridpoint, push through
     the polynomial to get yourself into warped image coordinate (but not yet 
     lifted onto the sky).  Then, using the set of warped gridpoints as
     inputs, fit back to their original grid locations as targets.
     */

	int inv_sip_order, ngrid, inv_sip_coeffs;

	double maxu, maxv, minu, minv;
	int i, j, p, q, gu, gv;
	int N, M;
	double u, v, U, V;

	gsl_matrix *mA, *QR;
	gsl_vector *b1, *b2, *x1, *x2;

	assert(t->sip->a_order == t->sip->b_order);
	assert(t->sip->ap_order == t->sip->bp_order);

	inv_sip_order = t->sip->ap_order;

	// Number of grid points to use:
	ngrid = 10 * (inv_sip_order + 1);
	logverb("tweak inversion using %u gridpoints\n", ngrid);

	// We only compute the upper triangle polynomial terms, and we exclude the
	// 0,0 element.
	inv_sip_coeffs = (inv_sip_order + 1) * (inv_sip_order + 2) / 2 - 1;

	// Number of samples to fit.
	M = ngrid * ngrid;
	// Number of coefficients to solve for.
	N = inv_sip_coeffs;

	mA = gsl_matrix_alloc(M, N);
	b1 = gsl_vector_alloc(M);
	b2 = gsl_vector_alloc(M);
	x1 = gsl_vector_alloc(N);
	x2 = gsl_vector_alloc(N);
	assert(mA);
	assert(b1);
	assert(b2);
	assert(x1);
	assert(x2);

	// Rearranging formula (4), (5), and (6) from the SIP paper gives the
	// following equations:
	//
	//   +----------------------- Linear pixel coordinates in PIXELS
	//   |                        before SIP correction
	//   |                   +--- Intermediate world coordinates in DEGREES
	//   |                   |
	//   v                   v
	//                  -1
	//   U = [CD11 CD12]   * x
	//   V   [CD21 CD22]     y
	//
	//   +---------------- PIXEL distortion delta from telescope to
	//   |                 linear coordinates
	//   |    +----------- Linear PIXEL coordinates before SIP correction
	//   |    |       +--- Polynomial U,V terms in powers of PIXELS
	//   v    v       v
	//
	//   -f(u1,v1) =  p11 p12 p13 p14 p15 ... * ap1
	//   -f(u2,v2) =  p21 p22 p23 p24 p25 ...   ap2
	//   ...
	//
	//   -g(u1,v1) =  p11 p12 p13 p14 p15 ... * bp1
	//   -g(u2,v2) =  p21 p22 p23 p24 p25 ...   bp2
	//   ...
	//
	// which recovers the A and B's.

	// Find image boundaries
	minu = minv = 1e100;
	maxu = maxv = -1e100;

	for (i = 0; i < t->n; i++) {
		minu = MIN(minu, t->x[i] - t->sip->wcstan.crpix[0]);
		minv = MIN(minv, t->y[i] - t->sip->wcstan.crpix[1]);
		maxu = MAX(maxu, t->x[i] - t->sip->wcstan.crpix[0]);
		maxv = MAX(maxv, t->y[i] - t->sip->wcstan.crpix[1]);
	}

	// Sample grid locations.
	i = 0;
	for (gu = 0; gu < ngrid; gu++) {
		for (gv = 0; gv < ngrid; gv++) {
			double fuv, guv;
			// Calculate grid position in original image pixels
			u = (gu * (maxu - minu) / (ngrid-1)) + minu; // now in pixels
			v = (gv * (maxv - minv) / (ngrid-1)) + minv;  // now in pixels
			// compute U=u+f(u,v) and V=v+g(u,v)
			sip_calc_distortion(t->sip, u, v, &U, &V);
			fuv = U - u;
			guv = V - v;

			// Polynomial terms...
			j = 0;
			for (p = 0; p <= inv_sip_order; p++)
				for (q = 0; q <= inv_sip_order; q++)
					if ((p + q > 0) &&
						(p + q <= inv_sip_order)) {
						assert(j < N);
						gsl_matrix_set(mA, i, j, pow(U, (double)p) * pow(V, (double)q));
						j++;
					}
			assert(j == N);
			gsl_vector_set(b1, i, -fuv);
			gsl_vector_set(b2, i, -guv);
			i++;
		}
	}

	// Solve the linear equation.
	{
		double rmsB=0;
		gsl_vector *tau, *resid1, *resid2;
		int ret;

		tau = gsl_vector_alloc(imin(M, N));
		resid1 = gsl_vector_alloc(M);
		resid2 = gsl_vector_alloc(M);
		assert(tau);
		assert(resid1);
		assert(resid2);

		ret = gsl_linalg_QR_decomp(mA, tau);
		assert(ret == 0);
		// mA,tau now contains a packed version of Q,R.
		QR = mA;

		ret = gsl_linalg_QR_lssolve(QR, tau, b1, x1, resid1);
		assert(ret == 0);
		ret = gsl_linalg_QR_lssolve(QR, tau, b2, x2, resid2);
		assert(ret == 0);

		// RMS of (AX-B).
		for (j=0; j<M; j++) {
			rmsB += square(gsl_vector_get(resid1, j));
			rmsB += square(gsl_vector_get(resid2, j));
		}
		if (M > 0)
			rmsB = sqrt(rmsB / (double)(M*2));
		debug("gsl rms                = %g\n", rmsB);

		gsl_vector_free(tau);
		gsl_vector_free(resid1);
		gsl_vector_free(resid2);
	}

	// Extract the coefficients
	j = 0;
	for (p = 0; p <= inv_sip_order; p++)
		for (q = 0; q <= inv_sip_order; q++)
			if ((p + q > 0) &&
				(p + q <= inv_sip_order)) {
				assert(j < N);
				t->sip->ap[p][q] = gsl_vector_get(x1, j);
				t->sip->bp[p][q] = gsl_vector_get(x2, j);
				j++;
			}
	assert(j == N);

	// Check that we found values that actually invert the polynomial.
	// The error should be particularly small at the grid points.
	if (log_get_level() > LOG_VERB) {
		// rms error accumulators:
		double sumdu = 0;
		double sumdv = 0;
		int Z;

		for (gu = 0; gu < ngrid; gu++) {
			for (gv = 0; gv < ngrid; gv++) {
				double newu, newv;
				// Calculate grid position in original image pixels
				u = (gu * (maxu - minu) / (ngrid-1)) + minu;
				v = (gv * (maxv - minv) / (ngrid-1)) + minv;
				sip_calc_distortion(t->sip, u, v, &U, &V);
				sip_calc_inv_distortion(t->sip, U, V, &newu, &newv);
				sumdu += square(u - newu);
				sumdv += square(v - newv);
			}
		}
		sumdu /= (ngrid*ngrid);
		sumdv /= (ngrid*ngrid);
		debug("RMS error of inverting a distortion (at the grid points):\n");
		debug("  du: %g\n", sqrt(sumdu));
		debug("  dv: %g\n", sqrt(sumdu));
		debug("  dist: %g\n", sqrt(sumdu + sumdv));

		sumdu = 0;
		sumdv = 0;
		Z = 1000;
		for (i=0; i<Z; i++) {
			double newu, newv;
			u = uniform_sample(minu, maxu);
			v = uniform_sample(minv, maxv);
			sip_calc_distortion(t->sip, u, v, &U, &V);
			sip_calc_inv_distortion(t->sip, U, V, &newu, &newv);
			sumdu += square(u - newu);
			sumdv += square(v - newv);
		}
		sumdu /= Z;
		sumdv /= Z;
		debug("RMS error of inverting a distortion (at random points):\n");
		debug("  du: %g\n", sqrt(sumdu));
		debug("  dv: %g\n", sqrt(sumdu));
		debug("  dist: %g\n", sqrt(sumdu + sumdv));
	}

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
void do_sip_tweak(tweak_t* t) {
	int sip_order, sip_coeffs, stride;
	double xyzcrval[3];
	double cdinv[2][2];
	double sx, sy, sU, sV, su, sv;
	sip_t* swcs;
	int M, N;
	int i, j, p, q, order;
	int row;
	double totalweight;

	gsl_matrix *mA, *QR;
	gsl_vector *b1, *b2, *x1, *x2;

	// a_order and b_order should be the same!
	assert(t->sip->a_order == t->sip->b_order);
	sip_order = t->sip->a_order;

	// We need at least the linear terms to compute CD.
	if (sip_order < 1)
		sip_order = 1;

	// The SIP coefficients form an (order x order) upper triangular
	// matrix missing the 0,0 element.
	sip_coeffs = (sip_order + 1) * (sip_order + 2) / 2;

	/* calculate how many points to use based on t->include */
	stride = 0;
	for (i = 0; i < il_size(t->included); i++)
		if (il_get(t->included, i))
			stride++;
	assert(il_size(t->included) == il_size(t->image));

	M = stride;
	N = sip_coeffs;

    if (M < N) {
        logmsg("Too few correspondences for the SIP order specified (%i < %i)\n", M, N);
        return;
    }

	mA = gsl_matrix_alloc(M, N);
	b1 = gsl_vector_alloc(M);
	b2 = gsl_vector_alloc(M);
	x1 = gsl_vector_alloc(N);
	x2 = gsl_vector_alloc(N);
	assert(mA);
	assert(b1);
	assert(b2);
	assert(x1);
	assert(x2);

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
     * 
     *  We want to solve:
     * 
     *     min || b[M-by-1] - A[M-by-N] x[N-by-1] ||_2
     * 
     *  M = the number of correspondences.
     *  N = the number of SIP terms.
     *
     * And we want an overdetermined system, so M >= N.
     * 
     *           [ 1  u1   v1  u1^2  u1v1  v1^2  ... ]
     *    mA  =  [ 1  u2   v2  u2^2  u2v2  v2^2  ... ]
     *           [           ......                  ]
     *
     *         [ x1 ]
     *    b1 = [ x2 ]
     *         [ ...]
     *
     *         [ y1 ]
     *    b2 = [ y2 ]
     *         [ ...]
     *
     *  And the answers are:
     *
     *         [ sx              ]
     *    x1 = [ cd11            ]
     *         [ cd12            ]
     *         [ cd11*A + cd12*B ]
     *
     *         [ sy              ]
     *    x2 = [ cd21            ]
     *         [ cd22            ]
     *         [ cd21*A + cd22*B ]
     *
     *  (where A and B are actually tall vectors)
     *
     */

	// Fill in matrix mA:
	radecdeg2xyzarr(t->sip->wcstan.crval[0], t->sip->wcstan.crval[1], xyzcrval);
	totalweight = 0.0;
	i = -1;
	for (row = 0; row < il_size(t->included); row++)
        {
            int refi;
            double x, y;
            double xyzpt[3];
            double weight = 1.0;
            double u;
            double v;
            bool ok;

            if (!il_get(t->included, row)) {
                continue;
            }
            i++;

            assert(i >= 0);
            assert(i < M);

            u = t->x[il_get(t->image, i)] - t->sip->wcstan.crpix[0];
            v = t->y[il_get(t->image, i)] - t->sip->wcstan.crpix[1];

            if (t->weighted_fit) {
                //double dist2 = dl_get(t->dist2, i);
                //exp(-dist2 / (2.0 * dist2sigma));
                weight = dl_get(t->weight, i);
                assert(weight >= 0.0);
                assert(weight <= 1.0);
                totalweight += weight;
            }

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

            //  p q
            // (0,0) = 1     <- order 0
            // (1,0) = u     <- order 1
            // (0,1) = v
            // (2,0) = u^2   <- order 2
            // (1,1) = uv
            // (0,2) = v^2
            // ...

            // The shift - aka (0,0) - SIP coefficient must be 1.
            assert(gsl_matrix_get(mA, i, 0) == 1.0 * weight);
            assert(fabs(gsl_matrix_get(mA, i, 1) - u * weight) < 1e-12);
            assert(fabs(gsl_matrix_get(mA, i, 2) - v * weight) < 1e-12);

            // B contains Intermediate World Coordinates (in degrees)
            refi = il_get(t->ref, i);
            radecdeg2xyzarr(t->a_ref[refi], t->d_ref[refi], xyzpt);
            ok = star_coords(xyzpt, xyzcrval, &y, &x); // tangent-plane projection
            assert(ok);

            gsl_vector_set(b1, i, weight * rad2deg(x));
            gsl_vector_set(b2, i, weight * rad2deg(y));
        }
	assert(i == M - 1);

	if (t->weighted_fit)
		logverb("Total weight: %g\n", totalweight);

	// Solve the equation.
	{
		gsl_vector *tau, *resid1, *resid2;
		int ret;
		double rmsB=0;

		tau = gsl_vector_alloc(imin(M, N));
		resid1 = gsl_vector_alloc(M);
		resid2 = gsl_vector_alloc(M);
		assert(tau);
		assert(resid1);
		assert(resid2);

		ret = gsl_linalg_QR_decomp(mA, tau);
		assert(ret == 0);
		// mA,tau now contains a packed version of Q,R.
		QR = mA;

		ret = gsl_linalg_QR_lssolve(QR, tau, b1, x1, resid1);
		assert(ret == 0);
		ret = gsl_linalg_QR_lssolve(QR, tau, b2, x2, resid2);
		assert(ret == 0);

		// Find RMS of (AX - B)
		for (j=0; j<M; j++) {
			rmsB += square(gsl_vector_get(resid1, j));
			rmsB += square(gsl_vector_get(resid2, j));
		}
		if (M > 0)
			rmsB = sqrt(rmsB / (double)(M*2));
		logverb("gsl rms                = %g\n", rmsB);

		gsl_vector_free(tau);
		gsl_vector_free(resid1);
		gsl_vector_free(resid2);
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

	sU =
		cdinv[0][0] * sx +
		cdinv[0][1] * sy;
	sV =
		cdinv[1][0] * sx +
		cdinv[1][1] * sy;

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

	//	sU = get(X, 2, 0);
	//	sV = get(X, 2, 1);
	sip_calc_inv_distortion(t->sip, sU, sV, &su, &sv);
	//	su *= -1;
	//	sv *= -1;
	//printf("sU=%g, su=%g, sV=%g, sv=%g\n", sU, su, sV, sv);
	//printf("before cdinv b0=%g, b1=%g\n", get(b, 2, 0), get(b, 2, 1));
	//printf("BEFORE crval=(%.12g,%.12g)\n", t->sip->wcstan.crval[0], t->sip->wcstan.crval[1]);

	logverb("sx = %g, sy = %g\n", sx, sy);
	logverb("sU = %g, sV = %g\n", sU, sV);
	logverb("su = %g, sv = %g\n", su, sv);

	/*
     printf("Before applying shift:\n");
     sip_print_to(t->sip, stdout);
     printf("RMS error of correspondences: %g arcsec\n",
     correspondences_rms_arcsec(t));
     */

	swcs = wcs_shift(t->sip, -su, -sv);
	memcpy(t->sip, swcs, sizeof(sip_t));
	sip_free(swcs);

	//sip_free(t->sip);
	//t->sip = swcs;

	/*
     t->sip->wcstan.crpix[0] -= su;
     t->sip->wcstan.crpix[1] -= sv;
     */

	logverb("After applying shift:\n");
    if (t->verbose)
        sip_print_to(t->sip, stdout);

	/*
     printf("shiftxun=%g, shiftyun=%g\n", sU, sV);
     printf("shiftx=%g, shifty=%g\n", su, sv);
     printf("sqerr=%g\n", figure_of_merit(t,NULL,NULL));
     */

	// this data is now wrong
	tweak_clear_image_ad(t);
	tweak_clear_ref_xy(t);
	tweak_clear_image_xyz(t);

	// recalc based on new SIP
	tweak_go_to(t, TWEAK_HAS_IMAGE_AD);
	tweak_go_to(t, TWEAK_HAS_REF_XY);

	/*
     printf("+++++++++++++++++++++++++++++++++++++\n");
     printf("RMS=%g [arcsec on sky]\n", sqrt(figure_of_merit(t,NULL,NULL) / stride));
     printf("+++++++++++///////////+++++++++++++++\n");
     //	fprintf(stderr,"sqerrxy=%g\n", figure_of_merit2(t));
     */

	logverb("RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 0));
	logverb("Weighted RMS error of correspondences: %g arcsec\n",
            correspondences_rms_arcsec(t, 1));


	//	t->sip->wcstan.cd[0][0] = tmptan.cd[0][0];
	//	t->sip->wcstan.cd[0][1] = tmptan.cd[0][1];
	//	t->sip->wcstan.cd[1][0] = tmptan.cd[1][0];
	//	t->sip->wcstan.cd[1][1] = tmptan.cd[1][1];
	//	t->sip->wcstan.crval[0] = tmptan.crval[0];
	//	t->sip->wcstan.crval[1] = tmptan.crval[1];
	//	t->sip->wcstan.crpix[0] = tmptan.crpix[0];
	//	t->sip->wcstan.crpix[1] = tmptan.crpix[1];

	gsl_matrix_free(mA);
	gsl_vector_free(b1);
	gsl_vector_free(b2);
	gsl_vector_free(x1);
	gsl_vector_free(x2);
}

// RANSAC from Wikipedia:
// Given:
//     data - a set of observed data points
//     model - a model that can be fitted to data points
//     n - the minimum number of data values required to fit the model
//     k - the maximum number of iterations allowed in the algorithm
//     t - a threshold value for determining when a data point fits a model
//     d - the number of close data values required to assert that a model fits well to data
// Return:
//     bestfit - model parameters which best fit the data (or nil if no good model is found)
// iterations = 0
// bestfit = nil
// besterr = something really large
// while iterations < k {
//     maybeinliers = n randomly selected values from data
//     maybemodel = model parameters fitted to maybeinliers
//     alsoinliers = empty set
//     for every point in data not in maybeinliers {
//         if point fits maybemodel with an error smaller than t
//              add point to alsoinliers
//     }
//     if the number of elements in alsoinliers is > d {
//         % this implies that we may have found a good model
//         % now test how good it is
//         bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
//         thiserr = a measure of how well model fits these points
//         if thiserr < besterr {
//             bestfit = bettermodel
//             besterr = thiserr
//         }
//     }
//     increment iterations
// }
// return bestfit

void do_ransac(tweak_t* t)
{
	int iterations = 0;
	int maxiter = 500;

	sip_t wcs_try, wcs_best;

	double besterr = 100000000000000.;
	int sorder = t->sip->a_order;
	int num_free_coeffs = sorder*(sorder+1) + 4 + 2; // CD and CRVAL
	int min_data_points = num_free_coeffs/2 + 5;
	int set_size;
	il* maybeinliers;
	il* used_ref_sources;
	il* used_image_sources;
	int i;

    //	min_data_points *= 2;
    //	min_data_points *= 2;
	memcpy(&wcs_try, t->sip, sizeof(sip_t));
	memcpy(&wcs_best, t->sip, sizeof(sip_t));
	printf("/--------------------\n");
	printf("&&&&&&&&&&& mindatapts=%d\n", min_data_points);
	printf("\\-------------------\n");
	set_size = il_size(t->image);
	assert( il_size(t->image) == il_size(t->ref) );
	maybeinliers = il_new(30);
    //	il* alsoinliers = il_new(4);

	// we need to prevent pairing any reference star to multiple image
	// stars, or multiple reference stars to single image stars
	used_ref_sources = il_new(t->n_ref);
	used_image_sources = il_new(t->n);

	for (i=0; i<t->n_ref; i++) 
		il_append(used_ref_sources, 0);
	for (i=0; i<t->n; i++) 
		il_append(used_image_sources, 0);

	while (iterations++ < maxiter) {
		double thiserr;
		//		assert(t->ref);
		printf("++++++++++ ITERATION %d\n", iterations);

		// select n random pairs to use for the fit
		il_remove_all(maybeinliers);
		for (i=0; i<t->n_ref; i++) 
			il_set(used_ref_sources, i, 0);
		for (i=0; i<t->n; i++) 
			il_set(used_image_sources, i, 0);
		while (il_size(maybeinliers) < min_data_points) {
			int r = rand()/(double)RAND_MAX * set_size;
            //			printf("eeeeeeeeeeeeeeeee %d\n", r);
			// check to see if either star in this pairing is
			// already taken before adding this pairing
			int ref_ind = il_get(t->ref, r);
			int image_ind = il_get(t->image, r);
			if (!il_get(used_ref_sources, ref_ind) &&
			    !il_get(used_image_sources, image_ind)) {
				il_insert_unique_ascending(maybeinliers, r);
				il_set(used_ref_sources, ref_ind, 1);
				il_set(used_image_sources, image_ind, 1);
			}
		}
		for (i=0; i<il_size(t->included); i++) 
			il_set(t->included, i, 0);
		for (i=0; i<il_size(maybeinliers); i++) 
			il_set(t->included, il_get(maybeinliers,i), 1);
		
		// now do a fit with our random sample selection
		t->state &= ~TWEAK_HAS_LINEAR_CD;
		tweak_go_to(t, TWEAK_HAS_LINEAR_CD);

		// this data is now wrong
		tweak_clear_image_ad(t);
		tweak_clear_ref_xy(t);
		tweak_clear_image_xyz(t);

		// recalc based on new SIP
		tweak_go_to(t, TWEAK_HAS_IMAGE_AD);
		tweak_go_to(t, TWEAK_HAS_REF_XY);
		tweak_go_to(t, TWEAK_HAS_IMAGE_XYZ);

		// rms arcsec
		thiserr = sqrt(figure_of_merit(t,NULL,NULL) / il_size(t->ref));
		if (thiserr < besterr) {
			besterr = thiserr;
		}

		/*
         // now find other samples which do well under the model fit by
         // the random sample set.
         il_remove_all(alsoinliers);
         for (i=0; i<il_size(t->included); i++) {
         if (il_get(t->included, i))
         continue;
         double thresh = 2.e-04; // FIXME mystery parameter
         double image_xyz[3];
         double ref_xyz[3];
         int ref_ind = il_get(t->ref, i);
         int image_ind = il_get(t->image, i);
         double a,d;
         pixelxy2radec(t->sip, t->x[image_ind],t->x[image_ind], &a,&d);
         radecdeg2xyzarr(a,d,image_xyz);
         radecdeg2xyzarr(t->a_ref[ref_ind],t->d_ref[ref_ind],ref_xyz);
         double dx = ref_xyz[0] - image_xyz[0];
         double dy = ref_xyz[1] - image_xyz[1];
         double dz = ref_xyz[2] - image_xyz[2];
         double err = dx*dx+dy*dy+dz*dz;
         if (sqrt(err) < thresh)
         il_append(alsoinliers, i);
         }

         // if we found a good number of points which are really close,
         // then fit both our random sample and the other close points
         if (10 < il_size(alsoinliers)) { // FIXME mystery parameter

         printf("found extra samples %d\n", il_size(alsoinliers));
         for (i=0; i<il_size(alsoinliers); i++) 
         il_set(t->included, il_get(alsoinliers,i), 1);
			
         // FIT AGAIN
         // FIXME put tweak here
         if (t->err < besterr) {
         memcpy(&wcs_best, t->sip, sizeof(sip_t));
         besterr = t->err;
         printf("new best error %g\n", besterr);
         }
         }
         printf("error=%g besterror=%g\n", t->err, besterr);
         */
	}
	printf("==============================\n");
	printf("==============================\n");
	printf("besterr = %g \n", besterr);
	printf("==============================\n");
	printf("==============================\n");
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

unsigned int tweak_advance_to(tweak_t* t, unsigned int flag)
{
	//	fprintf(stderr,"WANT: ");
	//	tweak_print_the_state(flag);
	//	fprintf(stderr,"\n");
	want(TWEAK_HAS_IMAGE_AD) {
		//printf("////++++-////\n");
		int jj;
		ensure(TWEAK_HAS_SIP);
		ensure(TWEAK_HAS_IMAGE_XY);

		logverb("Satisfying TWEAK_HAS_IMAGE_AD\n");

		// Convert to ra dec
		assert(!t->a);
		assert(!t->d);
		t->a = malloc(sizeof(double) * t->n);
		t->d = malloc(sizeof(double) * t->n);
		for (jj = 0; jj < t->n; jj++) {
			sip_pixelxy2radec(t->sip, t->x[jj], t->y[jj], t->a + jj, t->d + jj);
			//			printf("i=%4d, x=%10g, y=%10g ==> a=%10g, d=%10g\n", jj, t->x[jj], t->y[jj], t->a[jj], t->d[jj]);
		}

		done(TWEAK_HAS_IMAGE_AD);
	}

	want(TWEAK_HAS_REF_AD) {
		ensure(TWEAK_HAS_REF_XYZ);

		// FIXME
		logverb("Satisfying TWEAK_HAS_REF_AD\n");
		logerr("FIXME!\n");

		done(TWEAK_HAS_REF_AD);
	}

	want(TWEAK_HAS_REF_XY) {
		int jj;
		ensure(TWEAK_HAS_REF_AD);

		//tweak_clear_ref_xy(t);

		logverb("Satisfying TWEAK_HAS_REF_XY\n");
		assert(t->state & TWEAK_HAS_REF_AD);
		assert(t->n_ref);
		assert(!t->x_ref);
		assert(!t->y_ref);
		t->x_ref = malloc(sizeof(double) * t->n_ref);
		t->y_ref = malloc(sizeof(double) * t->n_ref);
		for (jj = 0; jj < t->n_ref; jj++) {
			bool ok;
			ok = sip_radec2pixelxy(t->sip, t->a_ref[jj], t->d_ref[jj],
								   t->x_ref + jj, t->y_ref + jj);
			assert(ok);
			//fprintf(stderr,"ref star %04d: %g,%g\n",jj,t->x_ref[jj],t->y_ref[jj]);
		}

		done(TWEAK_HAS_REF_XY);
	}

	want(TWEAK_HAS_AD_BAR_AND_R) {
		ensure(TWEAK_HAS_IMAGE_AD);

		logverb("Satisfying TWEAK_HAS_AD_BAR_AND_R\n");

		assert(t->state & TWEAK_HAS_IMAGE_AD);
		get_center_and_radius(t->a, t->d, t->n,
		                      &t->a_bar, &t->d_bar, &t->radius);
		logverb("a_bar=%g [deg], d_bar=%g [deg], radius=%g [arcmin]\n",
		        t->a_bar, t->d_bar, rad2arcmin(t->radius));

		done(TWEAK_HAS_AD_BAR_AND_R);
	}

	want(TWEAK_HAS_IMAGE_XYZ) {
		int i;
		ensure(TWEAK_HAS_IMAGE_AD);

		logverb("Satisfying TWEAK_HAS_IMAGE_XYZ\n");
		assert(!t->xyz);

		t->xyz = malloc(3 * t->n * sizeof(double));
		for (i = 0; i < t->n; i++) {
			radecdeg2xyzarr(t->a[i], t->d[i], t->xyz + 3*i);
		}
		done(TWEAK_HAS_IMAGE_XYZ);
	}

	want(TWEAK_HAS_REF_XYZ) {
        logverb("Satisfying TWEAK_HAS_REF_XYZ\n");
        ensure(TWEAK_HAS_REF_AD);
        tweak_ref_find_xyz_from_ad(t);
		done(TWEAK_HAS_REF_XYZ);
	}

	want(TWEAK_HAS_COARSLY_SHIFTED) {
		ensure(TWEAK_HAS_REF_XY);
		ensure(TWEAK_HAS_IMAGE_XY);
		logverb("Satisfying TWEAK_HAS_COARSLY_SHIFTED\n");
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
		logverb("Satisfying TWEAK_HAS_FINELY_SHIFTED\n");
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
		logverb("Satisfying TWEAK_HAS_REALLY_FINELY_SHIFTED\n");
		// Shrink size of hough box
		do_entire_shift_operation(t, 0.03);
		tweak_clear_image_ad(t);
		tweak_clear_ref_xy(t);
		done(TWEAK_HAS_REALLY_FINELY_SHIFTED);
	}

	want(TWEAK_HAS_CORRESPONDENCES) {
		ensure(TWEAK_HAS_REF_XYZ);
		ensure(TWEAK_HAS_IMAGE_XYZ);
		logverb("Satisfying TWEAK_HAS_CORRESPONDENCES\n");
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
		logverb("Satisfying TWEAK_HAS_LINEAR_CD\n");
		do_sip_tweak(t);
		tweak_clear_on_sip_change(t);
		done(TWEAK_HAS_LINEAR_CD);
	}

	want(TWEAK_HAS_RUN_RANSAC_OPT) {
		ensure(TWEAK_HAS_CORRESPONDENCES);
		do_ransac(t);
		done(TWEAK_HAS_RUN_RANSAC_OPT);
	}

	logerr("die for dependence: ");
	tweak_print_the_state(flag);
	printf("\n");
	assert(0);
	return -1;
}

void tweak_push_wcs_tan(tweak_t* t, tan_t* wcs) {
	if (!t->sip)
		t->sip = sip_create();
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
	il_free(t->maybeinliers);
	il_free(t->bestinliers);
	il_free(t->included);
	t->image = NULL;
	t->ref = NULL;
	t->dist2 = NULL;
	t->weight = NULL;
	t->maybeinliers = NULL;
	t->bestinliers = NULL;
	t->included = NULL;
	kdtree_free(t->kd_image);
	kdtree_free(t->kd_ref);
}

void tweak_free(tweak_t* t) {
	tweak_clear(t);
	free(t);
}
