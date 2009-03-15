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
#include <assert.h>

#include "dimage.h"
#include "simplexy-common.h"
#include "log.h"

/*
 * dsigma.c
 *
 * Simple guess at the sky sigma
 *
 * Mike Blanton
 * 1/2006 */


int dsigma(float *image,
           int nx,
           int ny,
           int sp,
           int gridsize,
           float *sigma)
{
	float *diff = NULL;
	float tot;
	int i, j, n, dx, dy, ndiff;
    int rtn = 0;

	if (nx == 1 && ny == 1) {
		*sigma = 0.;
		return rtn;
	}

    if (gridsize == 0)
        gridsize = 50;

	dx = gridsize;
	if (dx > nx / 4)
		dx = nx / 4;
	if (dx <= 0)
		dx = 1;

	dy = gridsize;
	if (dy > ny / 4)
		dy = ny / 4;
	if (dy <= 0)
		dy = 1;

	/* get a bunch of noise 'samples' by looking at the differences between two
	 * diagonally spaced pixels (usually 5) */
    ndiff = ((nx-sp + dx-1)/dx) * ((ny-sp + dy-1)/dy);

	if (ndiff <= 1) {
		*sigma = 0.;
        return rtn;
	}

	diff = malloc(ndiff * sizeof(float));
	n = 0;
	for (j = 0; j < ny-sp; j += dy) {
		for (i = 0; i < nx-sp; i += dx) {
			diff[n] = fabs(image[i + j * nx] - image[i + sp + (j + sp) * nx]);
			n++;
		}
	}
    assert(n == ndiff);

	if (ndiff <= 10) {
		tot = 0.;
		for (i = 0; i < ndiff; i++)
			tot += diff[i] * diff[i];
		*sigma = sqrt(tot / (float) ndiff);
        goto finish;
	}

	/* estimate sigma in a clever way to avoid having our estimate biased by
	 * outliers. outliers come into the diff list when we sampled a point where
	 * the upper point was on a source, but the lower one was not (or vice
     * versa).  Since the
	 * sample variance involves squaring the already-large outliers, they
	 * drastically affect the final sigma estimate. by sorting, the outliers go
	 * to the top and only affect the final value very slightly, because they
	 * are a small fraction of the total entries in diff (or so we hope!) */

    {
		double Nsigma=0.7;
		double s = 0.0;
		while (s == 0.0) {
			int k = (int)floor(ndiff * erf(Nsigma/sqrt(2.0)));
			if (k == ndiff) {
				logerr("Failed to estimate the image noise.  Setting sigma=1.  Expect the worst.\n");
			    s = 1.0;
				break;
			}
			s = dselip(k, ndiff, diff) / (Nsigma * sqrt(2.));
			logverb("Nsigma=%g, s=%g\n", Nsigma, s);
			Nsigma += 0.1;
		}
        *sigma = s;
    }

    if (*sigma == 0.0) {
        int nzero = 0;
        int NS = ndiff;
        tot = 0.0;
        logverb("outlier-rejecting sigma is zero.\n");
        for (i=0; i<ndiff; i++) {
            if (diff[i] == 0.0)
                nzero++;
        }
        logverb("%i of %i diffs are zero.\n", nzero, ndiff);
        if (nzero < (0.9 * ndiff)) {
            // omit the top 5% as outliers...
            NS = (0.95 * ndiff);
        }

        for (i=0; i<NS; i++)
            tot += diff[i]*diff[i];
        *sigma = sqrt(tot / (float)NS);
        logverb("set sigma=%g\n", *sigma);
        /*
         for (i=0; i<ndiff; i++) {
         if (diff[i] != 0)
         break;
         }
         logverb("first non-zero difference is index %i of %i\n", i, ndiff);
         */
    }
    rtn = 1;

 finish:
    FREEVEC(diff);
    return rtn;
} /* end dsigma */
