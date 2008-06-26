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

int simplexy2(simplexy_t* s) {
	int i;
    int nx = s->nx;
    int ny = s->ny;

    /* Exactly one of s->image and s->image_u8 should be non-NULL.*/
    assert(s->image || s->image_u8);
    assert(!s->image || !s->image_u8);

    logverb("simplexy: nx=%d, ny=%d\n", nx, ny);
    logverb("simplexy: dpsf=%f, plim=%f, dlim=%f, saddle=%f\n",
            s->dpsf, s->plim, s->dlim, s->saddle);
    logverb("simplexy: maxper=%d, maxnpeaks=%d, maxsize=%d, halfbox=%d\n",
            s->maxper, s->maxnpeaks, s->maxsize, s->halfbox);

	/* median smooth */
    if (s->image) {
        logverb("simplexy: median smoothing...\n");
        s->simage = malloc(nx * ny * sizeof(float));
        dmedsmooth(s->image, nx, ny, s->halfbox, s->simage);
        for (i=0; i<nx*ny; i++)
            s->simage[i] = s->image[i] - s->simage[i];

    } else {
        // u8 image
        logverb("simplexy: median smoothing...\n");
        if (MIN(nx,ny) < 2*s->halfbox+1) {
            s->halfbox = floor(((float)MIN(nx,ny) - 1.0) / 2.0);
        }
        assert(MIN(nx,ny) >= 2*s->halfbox+1);

        s->simage_u8 = malloc(nx * ny * sizeof(unsigned char));
        ctmf(s->image_u8, s->simage_u8, nx, ny, nx, nx, s->halfbox, 1, 512*1024);
	
        s->simage = malloc(nx * ny * sizeof(float));
        for (i=0; i<nx*ny; i++)
            s->simage[i] = (float)s->image_u8[i] - (float)s->simage_u8[i];
        FREEVEC(s->simage_u8);

        s->image = malloc(nx * ny * sizeof(float));
        for (i=0; i<nx*ny; i++)
            s->image[i] = s->image_u8[i];
    }

	/* determine an estimate of the noise in the image (sigma) assuming the
	 * noise is iid gaussian, by sampling at regular intervals, and comparing
	 * the difference between pixels separated by a 5-pixel diagonal gap. */
    logverb("simplexy: measuring image noise (sigma)...\n");
	dsigma(s->image, nx, ny, 5, 0, &(s->sigma));
    logverb("simplexy: found sigma=%g.\n", s->sigma);
    if (s->sigma == 0.0) {
        logverb("simplexy: re-estimating sigma with a finer grid...\n");
        dsigma(s->image, nx, ny, 5, 5, &(s->sigma));
        logverb("simplexy: found sigma=%g.\n", s->sigma);
        if (s->sigma == 0.0) {
            logverb("simplexy: re-estimating sigma with a finer grid...\n");
            dsigma(s->image, nx, ny, 5, 1, &(s->sigma));
            logverb("simplexy: found sigma=%g.\n", s->sigma);
        }
        /*
        double mn,mx;
        int nzero;
        // hmm...
        logverb("simplexy: estimated image noise (sigma) is zero.\n");
        mn =  HUGE_VAL;
        mx = -HUGE_VAL;
        nzero = 0;
        for (i=0; i<nx*ny; i++) {
            mn = MIN(mn, s->image[i]);
            mx = MAX(mx, s->image[i]);
            if (s->image[i] == 0.0)
                nzero++;
        }
        logverb("simplexy: image contains %i of %i (%.1f %%) zero values.\n",
                nzero, nx*ny, 100.0*nzero/(double)(nx*ny));
        // MAGIC
        s->sigma = (mx - mn) * 0.1;
        if (s->sigma == 0.0)
            s->sigma = 1.0;
        logverb("simplexy: image range is [%g, %g]; setting sigma to %g.\n",
                mn, mx, s->sigma);
         */
    }

	/* find objects */
	s->smooth = malloc(nx * ny * sizeof(float));
	s->oimage = malloc(nx * ny * sizeof(int));
    logverb("simplexy: finding objects...\n");
	dobjects(s->simage, s->smooth, nx, ny, s->dpsf, s->plim, s->oimage);
	FREEVEC(s->smooth);

    s->x = malloc(s->maxnpeaks * sizeof(float));
    s->y = malloc(s->maxnpeaks * sizeof(float));
	
	/* find all peaks within each object */
    logverb("simplexy: finding peaks...\n");
	dallpeaks(s->simage, nx, ny, s->oimage, s->x, s->y, &(s->npeaks), s->dpsf,
	          s->sigma, s->dlim, s->saddle, s->maxper, s->maxnpeaks, s->sigma, s->maxsize);
    logmsg("simplexy: found %i sources.\n", s->npeaks);
	FREEVEC(s->oimage);

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
        s->flux[i]       = s->simage[ix + iy * nx];
        s->background[i] = s->image [ix + iy * nx] - s->simage[ix + iy * nx];
    }

	FREEVEC(s->simage);

    // for u8 images, we allocate a temporary float image in s->image.
    if (s->image_u8)
        FREEVEC(s->image);

	return 1;
}

