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
#include <stdint.h>
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

#define IMGTYPE float
#define SUFFIX
#include "dsmooth.inc"
#undef SUFFIX
#undef IMGTYPE

#define IMGTYPE uint8_t
#define SUFFIX _u8
#include "dsmooth.inc"
#undef IMGTYPE
#undef SUFFIX

#define IMGTYPE int16_t
#define SUFFIX _i16
#include "dsmooth.inc"
#undef IMGTYPE
#undef SUFFIX


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

