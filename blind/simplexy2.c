/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Michael Blanton, Keir Mierle, David W. Hogg,
 Sam Roweis and Dustin Lang.
 
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

#include "simplexy2.h"
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
    qfitsdumper qoutimg;
	qfits_header* hdr;

    memset(&qoutimg, 0, sizeof(qoutimg));
    qoutimg.filename = fn;
    qoutimg.npix = nx * ny;
    qoutimg.ptype = PTYPE_FLOAT;
    qoutimg.fbuf = img;
    qoutimg.out_ptype = BPP_IEEE_FLOAT;

    // write header.
    hdr = fits_get_header_for_image(&qoutimg, nx, NULL);

	if (fits_write_header_and_image(hdr, &qoutimg)) {
		ERROR("Failed to write FITS image to file \"%s\"", fn);
		exit(-1);
	}
}

void simplexy2_set_u8_defaults(simplexy_t* i) {
    memset(i, 0, sizeof(simplexy_t));
    simplexy2_set_defaults(i);
    i->plim      = SIMPLEXY_U8_DEFAULT_PLIM;
    i->saddle    = SIMPLEXY_U8_DEFAULT_SADDLE;
}

void simplexy2_set_defaults(simplexy_t* i) {
    memset(i, 0, sizeof(simplexy_t));
    i->dpsf      = SIMPLEXY_DEFAULT_DPSF;
    i->plim      = SIMPLEXY_DEFAULT_PLIM;
    i->dlim      = SIMPLEXY_DEFAULT_DLIM;
    i->saddle    = SIMPLEXY_DEFAULT_SADDLE;
    i->maxper    = SIMPLEXY_DEFAULT_MAXPER;
    i->maxsize   = SIMPLEXY_DEFAULT_MAXSIZE;
    i->halfbox   = SIMPLEXY_DEFAULT_HALFBOX;
    i->maxnpeaks = SIMPLEXY_DEFAULT_MAXNPEAKS;
}

void simplexy2_free_contents(simplexy_t* s) {
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

int simplexy2(simplexy_t* s) {
	int i;
    int nx = s->nx;
    int ny = s->ny;
	float smoothsigma;
    float limit;
	uint8_t* mask;
	// Background-subtracted image.
	float* theimage = NULL;
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

	if (s->image_u8) {
		// Float image.
		s->image = malloc(nx * ny * sizeof(float));
		for (i=0; i<nx*ny; i++)
			s->image[i] = s->image_u8[i];
	}

	if (s->nobgsub)
		theimage = s->image;
	else {
		/* median smooth */
		if (s->image) {
			logverb("simplexy: median smoothing...\n");
			theimage = malloc(nx * ny * sizeof(float));
			dmedsmooth(s->image, nx, ny, s->halfbox, theimage);
			for (i=0; i<nx*ny; i++)
				theimage[i] = s->image[i] - theimage[i];

		} else {
			// u8 image: run faster ctmf() median-smoother.
			unsigned char* smoothed_u8;

			logverb("simplexy: median smoothing...\n");
			if (MIN(nx,ny) < 2*s->halfbox+1)
				s->halfbox = floor(((float)MIN(nx,ny) - 1.0) / 2.0);
			assert(MIN(nx,ny) >= 2*s->halfbox+1);

			smoothed_u8 = malloc(nx * ny * sizeof(unsigned char));
			ctmf(s->image_u8, smoothed_u8, nx, ny, nx, nx, s->halfbox, 1, 512*1024);

			// Background-subtracted image.
			theimage = malloc(nx * ny * sizeof(float));
			for (i=0; i<nx*ny; i++) {
				theimage[i] = s->image[i] - (float)smoothed_u8[i];
			}
			free(smoothed_u8);
		}

		if (s->bgsubimgfn) {
			logverb("Writing background-subtracted image \"%s\"\n", s->bgsubimgfn);
			write_fits_float_image(theimage, nx, ny, s->bgsubimgfn);
		}
		if (s->bgimgfn) {
			float* bgimg = malloc(nx * ny * sizeof(float));
			for (i=0; i<nx*ny; i++)
				bgimg[i] = s->image[i] - theimage[i];
			logverb("Writing background image \"%s\"\n", s->bgimgfn);
			write_fits_float_image(bgimg, nx, ny, s->bgimgfn);
			free(bgimg);
		}
	}

	smoothed = malloc(nx * ny * sizeof(float));

	/* smooth by the point spread function (the optimal detection
	   filter, since we assume a symmetric Gaussian PSF) */
	dsmooth2(theimage, nx, ny, s->dpsf, smoothed);

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
		float* maskedimg = malloc(nx * ny * sizeof(float));
		logverb("Writing masked image \"%s\"\n", s->maskimgfn);
		for (i=0; i<nx*ny; i++)
			maskedimg[i] = mask[i] * s->image[i];
		write_fits_float_image(maskedimg, nx, ny, s->maskimgfn);
		free(maskedimg);
	}

	/* find connected-components in the mask image. */
	ccimg = malloc(nx * ny * sizeof(int));
	dfind2_u8(mask, nx, ny, ccimg);
	FREEVEC(mask);

	// estimate the noise in the image (sigma)
	logverb("simplexy: measuring image noise (sigma)...\n");
	dsigma(s->image, nx, ny, 5, 0, &(s->sigma));
	logverb("simplexy: found sigma=%g.\n", s->sigma);

    s->x = malloc(s->maxnpeaks * sizeof(float));
    s->y = malloc(s->maxnpeaks * sizeof(float));
	
	/* find all peaks within each object */
    logverb("simplexy: finding peaks...\n");
	dallpeaks(theimage, nx, ny, ccimg, s->x, s->y, &(s->npeaks), s->dpsf,
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
		// "theimage" is background-subtracted.
        s->flux[i]       = theimage[ix + iy * nx];
        s->background[i] = s->image[ix + iy * nx] - theimage[ix + iy * nx];
    }

	if (!s->nobgsub)
		FREEVEC(theimage);
    // for u8 images, we allocate a temporary float image in s->image.
    if (s->image_u8)
        FREEVEC(s->image);

	return 1;
}

void simplexy2_clean_cache() {
	dselip_cleanup();
}

