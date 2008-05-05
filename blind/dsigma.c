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

#include "dimage.h"
#include "simplexy-common.h"

/*
 * dsigma.c
 *
 * Simple guess at the sky sigma
 *
 * Mike Blanton
 * 1/2006 */


float dselip(unsigned long k, unsigned long n, float *arr);

int dsigma(float *image,
           int nx,
           int ny,
           int sp,
           float *sigma)
{
	float *diff = NULL;
	float tot;
	int i, j, dx, dy, ndiff;
    int rtn = 0;

	if (nx == 1 && ny == 1) {
		*sigma = 0.;
		return 0;
	}

	dx = 50;
	if (dx > nx / 4)
		dx = nx / 4;
	if (dx <= 0)
		dx = 1;

	dy = 50;
	if (dy > ny / 4)
		dy = ny / 4;
	if (dy <= 0)
		dy = 1;

	/* get a bunch of noise 'samples' by looking at the differences between two
	 * diagonally spaced pixels (usually 5) */
	diff = malloc(2 * nx * ny * sizeof(float));
	ndiff = 0;
	for (j = 0; j < ny-sp; j += dy) {
		for (i = 0; i < nx-sp; i += dx) {
			diff[ndiff] = fabs(image[i + j * nx] - image[i + sp + (j + sp) * nx]);
			ndiff++;
		}
	}

	if (ndiff <= 1) {
		*sigma = 0.;
        goto finish;
	}

	if (ndiff <= 10) {
		tot = 0.;
		for (i = 0; i < ndiff; i++)
			tot += diff[i] * diff[i];
		*sigma = sqrt(tot / (float) ndiff);
        goto finish;
	}

	/* estimate sigma in a clever way to avoid having our estimate biased by
	 * outliers. outliers come into the diff list when we sampled a point where
	 * the upper point was on a source, but the lower one was not. since the
	 * sample variance involves squaring the already-large outliers, they
	 * drastically affect the final sigma estimate. by sorting, the outliers go
	 * to the top and only affect the final value very slightly, because they
	 * are a small fraction of the total entries in diff (or so we hope!) */
	*sigma = dselip((int) floor(ndiff * 0.68), ndiff, diff) / sqrt(2.);
    rtn = 1;

 finish:
    FREEVEC(diff);
    return rtn;
} /* end dsigma */
