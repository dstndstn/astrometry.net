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

#include "dimage.h"
#include "simplexy-common.h"
#include "log.h"

/*
 * dobjects.c
 *
 * Object detection
 *
 * Mike Blanton
 * 1/2006 */

typedef unsigned char u8;

int dobjects(float *image,
             float *smooth,
             int nx,
             int ny,
             float dpsf,
             float plim,
             int *objects)
{
	int i, j, ip, jp, ist, ind, jst, jnd;
	float limit, sigma;
	u8* mask;
    int flagged_one = 0;
    
	/* smooth by the point spread function  */
	dsmooth2(image, nx, ny, dpsf, smooth);

	/* check how much noise is left after running a median filter and a point
	 * spread smooth */
	dsigma(smooth, nx, ny, (int)(10*dpsf), 0, &sigma);

	/* limit is the threshold at which to pay attention to a pixel */
	limit = sigma * plim;

	/* This makes a mask which dfind uses when looking at the pixels; dfind
	 * ignores any pixels the mask flagged as uninteresting. */
	mask = calloc(nx * ny, sizeof(u8));
	for (j = 0;j < ny;j++) {
		jst = MAX(0, j - (long) (3 * dpsf));
		jnd = MIN(ny-1, j + (long) (3 * dpsf));

		for (i = 0;i < nx;i++) {
			/* this pixel looks interesting; note that we've already median
			 * subtracted so it's valid to directly compare the image value to
			 * sigma*significance without adding a mean */
			if (smooth[i + j*nx] <= limit)
                continue;

            flagged_one = 1;
            ist = MAX(0, i - (long) (3 * dpsf));
            ind = MIN(nx-1, i + (long) (3 * dpsf));

            /* now that we found a single interesting pixel, flag a box
             * around it so the object finding code will be able to
             * accurately estimate the center. */
            for (jp = jst; jp <= jnd; jp++)
                for (ip = ist; ip <= ind; ip++)
                    mask[ip + jp*nx] = 1;
        }
	}

    if (!flagged_one) {
        /* no pixels were masked - what parameter settings would cause at
         least one pixel to be masked? */
        float maxval = -HUGE_VAL;
        for (i=0; i<(nx*ny); i++)
            maxval = MAX(maxval, smooth[i]);
        logmsg("No pixels were marked as significant.\n"
               "  sigma = %g, plim = %g, limit = %g\n"
               "  max value in image = %g\n",
               sigma, plim, limit, maxval);
        FREEVEC(mask);
        return 0;
    }

	/* the mask now looks almost all black, except it's been painted with a
	 * bunch of white boxes (each box 6*point spread width) on parts of the
	 * image that have statistically significant 'events', aka sources. */

	/* now run connected component analysis to find and number each blob */
	dfind2_u8(mask, nx, ny, objects);

	FREEVEC(mask);

	return 1;
} /* end dobjects */
