/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg, Sam Roweis
  and Dustin Lang.

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

#include "simplexy-common.h"

/*
 * dsmooth.c
 *
 * Smooth an image
 *
 * Mike Blanton
 * 1/2006 
 */

// Optimize version of dsmooth, with a separated Gaussian convolution.
void dsmooth2(float *image,
            int nx,
            int ny,
            float sigma,
            float *smooth)
{
	int i, j, npix, half, start, end, sample;
	float neghalfinvvar, total, scale, dx, sum;
	float* kernel1D;
    float* kernel_shifted;
	float* smooth_temp;

	// make the kernel
	npix = 2 * ((int) ceilf(3. * sigma)) + 1;
	half = npix / 2;
	kernel1D =  malloc(npix * sizeof(float));
	neghalfinvvar = -1.0 / (2.0 * sigma * sigma);
	for (i=0; i<npix; i++) {
        dx = ((float) i - 0.5 * ((float)npix - 1.));
        kernel1D[i] = exp((dx * dx) * neghalfinvvar);
	}

	// normalize the kernel
	total = 0.0;
	for (i=0; i<npix; i++)
        total += kernel1D[i];
	scale = 1. / total;
	for (i=0; i<npix; i++)
        kernel1D[i] *= scale;

	smooth_temp = malloc(sizeof(float) * MAX(nx, ny));

    // Here's some trickery: we set "kernel_shifted" to be an array where:
    //   kernel_shifted[0] is the middle of the array,
    //   kernel_shifted[-half] is the left edge (ie the first sample),
    //   kernel_shifted[half] is the right edge (last sample)
	kernel_shifted = kernel1D + half;

	// convolve in x direction, dumping results into smooth_temp
	for (j=0; j<ny; j++) {
        float* imagerow = image + j*nx;
        for (i=0; i<nx; i++) {
            /*
             The outer loops are over OUTPUT pixels;
             the "sample" loop is over INPUT pixels.

             We're summing over the input pixels that contribute to the value
             of the output pixel.
             */
            start = i - half;
            start = MAX(start, 0);
            end = i + half;
            end = MIN(end, nx-1);
            sum = 0.0;
            for (sample=start; sample <= end; sample++)
                sum += imagerow[sample] * kernel_shifted[sample - i];
            smooth_temp[i] = sum;
        }
        memcpy(smooth + j*nx, smooth_temp, nx * sizeof(float));
    }

	// convolve in the y direction, dumping results into smooth
	for (i=0; i<nx; i++) {
        float* imagecol = smooth + i;
        for (j=0; j<ny; j++) {
            start = j - half;
            start = MAX(start, 0);
            end = j + half;
            end = MIN(end, ny-1);
            sum = 0.0;
            for (sample=start; sample<=end; sample++)
                sum += imagecol[sample*nx] * kernel_shifted[sample - j];
            smooth_temp[j] = sum;
        }
        for (j=0; j<ny; j++)
            smooth[i + j*nx] = smooth_temp[j];
	}

	FREEVEC(smooth_temp);
	FREEVEC(kernel1D);
} /* end dsmooth2 */


// Original version of dsmooth, non-separated kernel.
int dsmooth(float *image,
            int nx,
            int ny,
            float sigma,
            float *smooth)
{
	int i, j, npix, half, ip, jp, ist, jst, isto, jsto, ind, jnd, ioff, joff;
	float invvar, total, scale, dx, dy;
	float* kernel;

	/* make kernel */
	npix = 2 * ((int) ceilf(3. * sigma)) + 1;
	half = npix / 2;
	kernel =  malloc(npix * npix * sizeof(float));
	invvar = 1. / sigma / sigma;
	for (i = 0;i < npix;i++)
		for (j = 0;j < npix;j++) {
			dx = ((float) i - 0.5 * ((float)npix - 1.));
			dy = ((float) j - 0.5 * ((float)npix - 1.));
			kernel[i + j*npix] = exp( -0.5 * (dx * dx + dy * dy) * invvar);
		}
	total = 0.;
	for (i = 0;i < npix;i++)
		for (j = 0;j < npix;j++)
			total += kernel[i + j * npix];
	scale = 1. / total;
	for (i = 0;i < npix;i++)
		for (j = 0;j < npix;j++)
			kernel[i + j*npix] *= scale;

	for (j = 0;j < ny;j++)
		for (i = 0;i < nx;i++)
			smooth[i + j*nx] = 0.;

	for (j = 0;j < ny;j++) {
		jsto = jst = j - half;
		jnd = j + half;
		if (jst < 0)
			jst = 0;
		if (jnd > ny - 1)
			jnd = ny - 1;
		for (i = 0;i < nx;i++) {
			isto = ist = i - half;
			ind = i + half;
			if (ist < 0)
				ist = 0;
			if (ind > nx - 1)
				ind = nx - 1;
			for (jp = jst;jp <= jnd;jp++)
				for (ip = ist;ip <= ind;ip++) {
					ioff = ip - isto;
					joff = jp - jsto;
					smooth[ip + jp*nx] += image[i + j * nx] *
					                      kernel[ioff + joff * npix];
				}
		}
	}

	FREEVEC(kernel);

	return (1);
} /* end photfrac */

