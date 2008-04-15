/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include <assert.h>

#include "ctmf.h"
#include "dimage.h"
#include "simplexy-common.h"
#include "cairoutils.h"

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

int simplexy_continue(float *image,
             int nx,
             int ny,
             float dpsf,    /* gaussian psf width; 1 is usually fine */
             float plim,    /* significance to keep; 8 is usually fine */
             float dlim,    /* closest two peaks can be; 1 is usually fine */
             float saddle,  /* saddle difference (in sig); 3 is usually fine */
             int maxper,    /* maximum number of peaks per object; 1000 */
             int maxnpeaks, /* maximum number of peaks total; 100000 */
             int maxsize,   /* maximum size for extended objects: 150 */
             float *sigma,
             float *x,
             float *y,
             float *flux,
             int *npeaks,
             int verbose,
	     float *simage) {
	int i;

	int *oimage = NULL;
	float *smooth = NULL;
	
	/* determine an estimate of the noise in the image (sigma) assuming the
	 * noise is iid gaussian, by sampling at regular intervals, and comparing
	 * the difference between pixels separated by a 5-pixel diagonal gap. */
	dsigma(image, nx, ny, 5, sigma);
	if (verbose)
	  fprintf(stderr, "simplexy: dsigma() found sigma=%g.\n", *sigma);

	/* find objects */
	smooth = (float *) malloc(nx * ny * sizeof(float));
	oimage = (int *) malloc(nx * ny * sizeof(int));
	dobjects(simage, smooth, nx, ny, dpsf, plim, oimage);
	if (verbose)
	  fprintf(stderr, "simplexy: finished dobjects().\n");
	FREEVEC(smooth);
	
	/* find all peaks within each object */
	dallpeaks(simage, nx, ny, oimage, x, y, npeaks, dpsf,
	          (*sigma), dlim, saddle, maxper, maxnpeaks, (*sigma), maxsize);
	if (verbose)
	  fprintf(stderr, "simplexy: dallpeaks() found %i peaks.\n", *npeaks);
	FREEVEC(oimage);

	for (i = 0;i < (*npeaks);i++)
		flux[i] = simage[((int) (x[i] + 0.5)) + ((int) (y[i] + 0.5)) * nx];
	FREEVEC(simage);

	return (1);
}


int simplexy(float *image,
             int nx,
             int ny,
             float dpsf,    /* gaussian psf width; 1 is usually fine */
             float plim,    /* significance to keep; 8 is usually fine */
             float dlim,    /* closest two peaks can be; 1 is usually fine */
             float saddle,  /* saddle difference (in sig); 3 is usually fine */
             int maxper,    /* maximum number of peaks per object; 1000 */
             int maxnpeaks, /* maximum number of peaks total; 100000 */
             int maxsize,   /* maximum size for extended objects: 150 */
             int halfbox,    /* size for sliding sky estimation box */
             float *sigma,
             float *x,
             float *y,
             float *flux,
             int *npeaks,
			 int verbose) {

	int i;
	float *simage = NULL;

	if (verbose) {
		fprintf(stderr, "simplexy: nx=%d, ny=%d\n", nx, ny);
		fprintf(stderr, "simplexy: dpsf=%f, plim=%f, dlim=%f, saddle=%f\n",
				dpsf, plim, dlim, saddle);
		fprintf(stderr, "simplexy: maxper=%d, maxnpeaks=%d, maxsize=%d, halfbox=%d\n",
				maxper, maxnpeaks, maxsize, halfbox);
	}

	/* median smooth */
	/* NB: over-write simage to save malloc */
	simage = (float *) malloc(nx * ny * sizeof(float));
	dmedsmooth(image, nx, ny, halfbox, simage);
	for (i=0; i<nx*ny; i++)
		simage[i] = image[i] - simage[i];
	if (verbose)
		fprintf(stderr, "simplexy: finished dmedsmooth().\n");

	return simplexy_continue(image, nx, ny, dpsf, plim, dlim, saddle, maxper,  
			maxnpeaks, maxsize, sigma, x, y, flux, npeaks, verbose,
			simage);
}

int simplexy_u8(unsigned char *image,
		int nx,
		int ny,
		float dpsf,    /* gaussian psf width; 1 is usually fine */
		float plim,    /* significance to keep; 8 is usually fine */
		float dlim,    /* closest two peaks can be; 1 is usually fine */
		float saddle,  /* saddle difference (in sig); 3 is usually fine */
		int maxper,    /* maximum number of peaks per object; 1000 */
		int maxnpeaks, /* maximum number of peaks total; 100000 */
		int maxsize,   /* maximum size for extended objects: 150 */
		int halfbox,    /* size for sliding sky estimation box */
		float *sigma,
		float *x,
		float *y,
		float *flux,
		int *npeaks,
		int verbose) {

	int i;
	float *simage = NULL;
	unsigned char *simage_u8 = NULL;
	float *fimage = NULL;

	if (verbose) {
		fprintf(stderr, "simplexy: nx=%d, ny=%d\n", nx, ny);
		fprintf(stderr, "simplexy: dpsf=%f, plim=%f, dlim=%f, saddle=%f\n",
				dpsf, plim, dlim, saddle);
		fprintf(stderr, "simplexy: maxper=%d, maxnpeaks=%d, maxsize=%d, halfbox=%d\n",
				maxper, maxnpeaks, maxsize, halfbox);
	}

	/* median smooth */
	if (MIN(nx,ny) < 2*halfbox+1) {
		halfbox = floor(((float)MIN(nx,ny) - 1.0) / 2.0);
	}
	assert(MIN(nx,ny) >= 2*halfbox+1);

	simage_u8 = (unsigned char *) malloc(nx * ny * sizeof(unsigned char));
	ctmf(image, simage_u8, nx, ny, nx, nx, halfbox, 1, 512*1024);
	
	/* it's so purdy */
	/*
	{
	unsigned char *simage_cairo = NULL;
	int c;
	simage_cairo = malloc(sizeof(unsigned char) * nx * ny * 4);
	for(i=0; i<nx*ny; i++) {
	for (c=0; c<3; c++)
	simage_cairo[4*i+c] = simage_u8[i];
	simage_cairo[4*i+3] = 255;
	}
	cairoutils_write_png("median.png", simage_cairo, nx, ny);
	}
	*/

	simage = (float *) malloc(nx * ny * sizeof(float));
	for (i=0; i<nx*ny; i++){
		simage[i] = (float)(image[i] - simage_u8[i]);
	}
	free(simage_u8);

	fimage = (float *) malloc(nx * ny * sizeof(float));
	for (i=0; i<nx*ny; i++){
		fimage[i] = (float)image[i];
	}

	if (verbose)
		fprintf(stderr, "simplexy: finished ctmf() median smoothing.\n");

	return simplexy_continue(fimage, nx, ny, dpsf, plim, dlim, saddle, maxper,  
			maxnpeaks, maxsize, sigma, x, y, flux, npeaks, verbose,
			simage);
}

