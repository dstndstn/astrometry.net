/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Michael Blanton, Keir Mierle, David W. Hogg,
 Sam Roweis and Dustin Lang.
 Copyright 2008, 2009 Dustin Lang.

 The Astrometry.net suite is free software; you can redistribute it
 and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free
 Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 02110-1301 USA
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include <assert.h>

#include "simplexy.h"
#include "ctmf.h"
#include "dimage.h"
#include "simplexy-common.h"
#include "log.h"
#include "errors.h"

/*
 * simplexy.c
 *
 * Find sources in a float image.
 *
 * Algorithm outline:
 * 1. Estimate image noise
 * 2. Median filter and subtract to eliminate low-frequence sky gradient
 * 3. Find statistically significant pixels
 *    - Mask those pixels and a box around them
 * 4. Do connected components analysis on the resulting mask to find each
 *    component (source)
 * 5. Find the peaks in each object
 *    For each object:
 *       - Find the objects boundary
 *       - Cut it out
 *       - Smooth it
 *       - Find peaks in resulting cutout
 *       - Chose the most representative peak
 * 6. Extract the flux of each object as the value of the image at the peak
 *
 * BUGS:
 *
 * Note: To make simplexy() reentrant, do the following:
 * #define SIMPLEXY_REENTRANT
 * Or compile all the simplexy files with -DSIMPLEXY_REENTRANT
 *
 * Mike Blanton
 * 1/2006
 *
 */

#include "fitsioutils.h"
#include "qfits.h"

static void write_fits_float_image(const float* img, int nx, int ny,
								   const char* fn) {
	if (fits_write_float_image(img, nx, ny, fn))
		exit(-1);
}

static void write_fits_u8_image(const uint8_t* img, int nx, int ny,
								const char* fn) {
	if (fits_write_u8_image(img, nx, ny, fn))
		exit(-1);
}

void simplexy_fill_in_defaults(simplexy_t* s) {
	if (s->dpsf == 0)
		s->dpsf = SIMPLEXY_DEFAULT_DPSF;
    if (s->plim == 0)
		s->plim = SIMPLEXY_DEFAULT_PLIM;
    if (s->dlim == 0)
		s->dlim = SIMPLEXY_DEFAULT_DLIM;
	if (s->saddle == 0)
		s->saddle = SIMPLEXY_DEFAULT_SADDLE;
    if (s->maxper == 0)
		s->maxper = SIMPLEXY_DEFAULT_MAXPER;
	if (s->maxsize == 0)
		s->maxsize = SIMPLEXY_DEFAULT_MAXSIZE;
	if (s->halfbox == 0)
		s->halfbox = SIMPLEXY_DEFAULT_HALFBOX;
	if (s->maxnpeaks == 0)
		s->maxnpeaks = SIMPLEXY_DEFAULT_MAXNPEAKS;
}

void simplexy_fill_in_defaults_u8(simplexy_t* s) {
	if (s->plim == 0)
		s->plim = SIMPLEXY_U8_DEFAULT_PLIM;
	if (s->saddle == 0)
		s->saddle = SIMPLEXY_U8_DEFAULT_SADDLE;
	simplexy_fill_in_defaults(s);
}

void simplexy_set_u8_defaults(simplexy_t* s) {
    memset(s, 0, sizeof(simplexy_t));
    simplexy_fill_in_defaults_u8(s);
}

void simplexy_set_defaults(simplexy_t* s) {
    memset(s, 0, sizeof(simplexy_t));
    simplexy_fill_in_defaults(s);
}

void simplexy_free_contents(simplexy_t* s) {
	free(s->image);
	s->image = NULL;
	free(s->image_u8);
	s->image_u8 = NULL;
	free(s->x);
	s->x = NULL;
	free(s->y);
	s->y = NULL;
	free(s->flux);
	s->flux = NULL;
	free(s->background);
	s->background = NULL;
}

int simplexy_run(simplexy_t* s) {
	int i;
    int nx = s->nx;
    int ny = s->ny;
	float smoothsigma;
    float limit;
	uint8_t* mask;
	// background-subtracted image.
	float* bgsub = NULL;
	uint8_t* bgsub_u8 = NULL;
	// PSF-smoothed image.
	float* smoothed = NULL;
	// Connected-components image.
	int* ccimg = NULL;
 
    /* Exactly one of s->image and s->image_u8 should be non-NULL.*/
    assert(s->image || s->image_u8);
    assert(!s->image || !s->image_u8);

    logverb("simplexy: nx=%d, ny=%d\n", nx, ny);
    logverb("simplexy: dpsf=%f, plim=%f, dlim=%f, saddle=%f\n",
            s->dpsf, s->plim, s->dlim, s->saddle);
    logverb("simplexy: maxper=%d, maxnpeaks=%d, maxsize=%d, halfbox=%d\n",
            s->maxper, s->maxnpeaks, s->maxsize, s->halfbox);

	if (s->nobgsub) {
		bgsub_u8 = s->image_u8;
		bgsub = s->image;

	} else {
		// background subtraction via median smoothing.
		logverb("simplexy: median smoothing...\n");

		if (s->image) {
			float* medianfiltered;
			medianfiltered = malloc(nx * ny * sizeof(float));
			dmedsmooth(s->image, nx, ny, s->halfbox, medianfiltered);

			if (s->bgimgfn) {
				logverb("Writing background (median-filtered) image \"%s\"\n", s->bgimgfn);
				write_fits_float_image(medianfiltered, nx, ny, s->bgimgfn);
			}

			// subtract background from image, placing result in background.
			for (i=0; i<nx*ny; i++)
				medianfiltered[i] = s->image[i] - medianfiltered[i];
			bgsub = medianfiltered;
			medianfiltered = NULL;

		} else {
			// u8 image: run faster ctmf() median-smoother.
			unsigned char* medianfiltered_u8;

			if (MIN(nx,ny) < 2*s->halfbox+1)
				s->halfbox = floor(((float)MIN(nx,ny) - 1.0) / 2.0);
			assert(MIN(nx,ny) >= 2*s->halfbox+1);

			medianfiltered_u8 = malloc(nx * ny * sizeof(unsigned char));
			ctmf(s->image_u8, medianfiltered_u8, nx, ny, nx, nx, s->halfbox, 1, 512*1024);

			if (s->bgimgfn) {
				logverb("Writing background (median-filtered) image \"%s\"\n", s->bgimgfn);
				write_fits_u8_image(medianfiltered_u8, nx, ny, s->bgimgfn);
			}

			// Background-subtracted image.
			bgsub_u8 = malloc(nx * ny);
			for (i=0; i<nx*ny; i++)
				bgsub_u8[i] = MAX(0, s->image_u8[i] - medianfiltered_u8[i]);
			free(medianfiltered_u8);
		}

		if (s->bgsubimgfn) {
			logverb("Writing background-subtracted image \"%s\"\n", s->bgsubimgfn);
			if (bgsub)
				write_fits_float_image(bgsub, nx, ny, s->bgsubimgfn);
			else
				write_fits_u8_image(bgsub_u8, nx, ny, s->bgsubimgfn);
		}
	}

	smoothed = malloc(nx * ny * sizeof(float));

	/* smooth by the point spread function (the optimal detection
	   filter, since we assume a symmetric Gaussian PSF) */
	if (bgsub)
		dsmooth2(bgsub, nx, ny, s->dpsf, smoothed);
	else
		dsmooth2_u8(bgsub_u8, nx, ny, s->dpsf, smoothed);

	/* measure the noise level in the psf-smoothed image. */
	dsigma(smoothed, nx, ny, (int)(10*s->dpsf), 0, &smoothsigma);
	logverb("simplexy: noise in smoothed image: %g\n", smoothsigma);

    logverb("simplexy: finding objects...\n");
	limit = smoothsigma * s->plim;

	/* find pixels above the noise level, and flag a box of pixels around each one. */
	mask = malloc(nx*ny);
	if (!dmask(smoothed, nx, ny, limit, s->dpsf, mask)) {
		free(smoothed);
		return 0;
	}
	FREEVEC(smoothed);

	/* save the mask image, if requested. */
	if (s->maskimgfn) {
		logverb("Writing masked image \"%s\"\n", s->maskimgfn);
		if (s->image_u8) {
			uint8_t* maskedimg = malloc(nx * ny);
			for (i=0; i<nx*ny; i++)
				maskedimg[i] = mask[i] * s->image_u8[i];
			write_fits_u8_image(maskedimg, nx, ny, s->maskimgfn);
			free(maskedimg);
		} else {
			float* maskedimg = malloc(nx * ny * sizeof(float));
			for (i=0; i<nx*ny; i++)
				maskedimg[i] = mask[i] * s->image[i];
			write_fits_float_image(maskedimg, nx, ny, s->maskimgfn);
			free(maskedimg);
		}
	}

	/* find connected-components in the mask image. */
	ccimg = malloc(nx * ny * sizeof(int));
	dfind2_u8(mask, nx, ny, ccimg);
	FREEVEC(mask);

	// estimate the noise in the image (sigma)
	logverb("simplexy: measuring image noise (sigma)...\n");
	if (s->image_u8)
		dsigma_u8(s->image_u8, nx, ny, 5, 0, &(s->sigma));
	else
		dsigma(s->image, nx, ny, 5, 0, &(s->sigma));
	logverb("simplexy: found sigma=%g.\n", s->sigma);

    s->x = malloc(s->maxnpeaks * sizeof(float));
    s->y = malloc(s->maxnpeaks * sizeof(float));
	
	/* find all peaks within each object */
    logverb("simplexy: finding peaks...\n");
	if (bgsub)
		dallpeaks(bgsub, nx, ny, ccimg, s->x, s->y, &(s->npeaks), s->dpsf,
				  s->sigma, s->dlim, s->saddle, s->maxper, s->maxnpeaks, s->sigma, s->maxsize);
	else
		dallpeaks_u8(bgsub_u8, nx, ny, ccimg, s->x, s->y, &(s->npeaks), s->dpsf,
					 s->sigma, s->dlim, s->saddle, s->maxper, s->maxnpeaks, s->sigma, s->maxsize);
    logmsg("simplexy: found %i sources.\n", s->npeaks);
	FREEVEC(ccimg);

    s->x   = realloc(s->x, s->npeaks * sizeof(float));
    s->y   = realloc(s->y, s->npeaks * sizeof(float));
    s->flux       = malloc(s->npeaks * sizeof(float));
    s->background = malloc(s->npeaks * sizeof(float));

	for (i = 0; i < s->npeaks; i++) {
        int ix = (int)(s->x[i] + 0.5);
        int iy = (int)(s->y[i] + 0.5);
        bool finite;
        finite = isfinite(s->x[i]);
        assert(finite);
        finite = isfinite(s->y[i]);
        assert(finite);
        assert(ix >= 0);
        assert(iy >= 0);
        assert(ix < nx);
        assert(iy < ny);
		if (bgsub) {
			s->flux[i]       = bgsub[ix + iy * nx];
			s->background[i] = s->image[ix + iy * nx] - bgsub[ix + iy * nx];
		} else {
			s->flux[i]       = bgsub_u8[ix + iy * nx];
			s->background[i] = s->image_u8[ix + iy * nx] - bgsub_u8[ix + iy * nx];
		}
    }

	if (!s->nobgsub) {
		FREEVEC(bgsub);
		FREEVEC(bgsub_u8);
	}

	return 1;
}

void simplexy_clean_cache() {
	dselip_cleanup();
}

