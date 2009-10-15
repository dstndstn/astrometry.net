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
#include <assert.h>
#include <sys/param.h>

#include "dimage.h"
#include "permutedsort.h"
#include "simplexy-common.h"

/*
 * dallpeaks.c
 *
 * Take image and list of objects, and produce list of all peaks (and
 * which object they are in).
 *
 * BUGS:
 *   - Returns no error analysis if the centroid sux.
 *   - Uses dead-reckon pixel center if dcen3x3 sux.
 *   - No out-of-memory checks
 *
 * Mike Blanton
 * 1/2006 */

/* Finds all peaks in the image by cutting a bounding box out around
 each one */

int dallpeaks(float *image,
              int nx,
              int ny,
              int *objects,
              float *xcen,
              float *ycen,
              int *npeaks,
              float dpsf,
              float sigma,
              float dlim,
              float saddle,
              int maxper,
              int maxnpeaks,
              float minpeak,
              int maxsize) {
	int i, j, l, current, nobj, oi, oj, xmax, ymax, xmin, ymin, onx, ony, nc, lobj;
	int xcurr, ycurr, imore ;
	float tmpxc, tmpyc, three[9];

	int *dobject = NULL;
	int *indx = NULL;
	float *oimage = NULL;
	float *simage = NULL;
	int npix = 0;
	int *xc = NULL;
	int *yc = NULL;

	/* Make a list which contains the indexes of each connected component next
	 * to each other. */
	/* FIXME this can be done muuuch smarter by running in two passes */
	/* first calc histogram; then fill in indx array. */
	indx = (int *) malloc(sizeof(int) * nx * ny);
	dobject = (int *) malloc(sizeof(int) * (nx * ny + 1));
	for (j = 0;j < ny;j++)
		for (i = 0;i < nx;i++)
			dobject[i + j*nx] = objects[i + j*nx];

    permutation_init(indx, nx*ny);
    permuted_sort(dobject, sizeof(int), compare_ints_asc, indx, nx*ny);

	/* skip over the uninformative pixels */
	for (l = 0;l < nx*ny && dobject[indx[l]] == -1; l++)
		;

	nobj = 0;
	*npeaks = 0;
	xc = (int *) malloc(sizeof(int) * maxper);
	yc = (int *) malloc(sizeof(int) * maxper);
	for (;l < nx*ny;) {
		current = dobject[indx[l]];

		/* get object limits */
		xmax = -1;
		xmin = nx + 1;
		ymax = -1;
		ymin = ny + 1;
		for (lobj = l;lobj < nx*ny && dobject[indx[lobj]] == current;lobj++) {
			xcurr = indx[lobj] % nx;
			ycurr = indx[lobj] / nx;
			xmin = MIN(xmin, xcurr);
			xmax = MAX(xmax, xcurr);
			ymin = MIN(ymin, ycurr);
			ymax = MAX(ymax, ycurr);
		}

		if (!(xmax - xmin > 2 && xmax - xmin < maxsize &&
			  ymax - ymin > 2 && ymax - ymin < maxsize &&
			  *npeaks < maxnpeaks))
			continue;

		/* make object cutout (if it is 3x3 or bigger) */
		onx = xmax - xmin + 1;
		ony = ymax - ymin + 1;

		if (onx * ony > npix) {
			free(oimage);
			free(simage);
			npix = onx * ony;
			oimage = malloc(npix * sizeof(float));
			simage = malloc(npix * sizeof(float));
		}

		for (oj=0; oj<ony; oj++)
			for (oi=0; oi<onx; oi++) {
				oimage[oi + oj*onx] = 0.;
				i = oi + xmin;
				j = oj + ymin;
				/* if the object number of this pixel matches the current
				 * object, then copy the pixel into our cutout. */
				if (dobject[i + j*nx] == nobj)
					oimage[oi + oj*onx] = image[i + j * nx];
			}

		/* find peaks in cutout */
		dsmooth2(oimage, onx, ony, dpsf, simage);
		dpeaks(simage, onx, ony, &nc, xc, yc,
			   sigma, dlim, saddle, maxper, 0, 1, minpeak);
		imore = 0;
		for (i = 0;i < nc;i++) {
			if (!(xc[i] > 0 && xc[i] < onx - 1 &&
				  yc[i] > 0 && yc[i] < ony - 1 &&
				  imore < (maxnpeaks - (*npeaks))))
				continue;

			/* install default centroid to begin */
			xcen[imore + (*npeaks)] = (float)(xc[i] + xmin);
			ycen[imore + (*npeaks)] = (float)(yc[i] + ymin);
			assert(isfinite(xcen[imore + *npeaks]));
			assert(isfinite(ycen[imore + *npeaks]));

			/* try to get centroid in the 3 x 3 box */
			for (oi=-1; oi<=1; oi++)
				for (oj=-1; oj<=1; oj++)
					three[oi + 1 + (oj + 1)*3] =
						simage[oi + xc[i] + (oj + yc[i]) * onx];
			if (dcen3x3(three, &tmpxc, &tmpyc)) {
				assert(isfinite(tmpxc));
				assert(isfinite(tmpyc));
				xcen[imore + (*npeaks)] = tmpxc
					+ (float)(xc[i] + xmin - 1);
				ycen[imore + (*npeaks)] = tmpyc
					+ (float)(yc[i] + ymin - 1);
				assert(isfinite(xcen[imore + *npeaks]));
				assert(isfinite(ycen[imore + *npeaks]));

			} else if (xc[i] > 1 && xc[i] < onx - 2 &&
					   yc[i] > 1 && yc[i] < ony - 2 &&
					   imore < (maxnpeaks - (*npeaks))) {
				/* try to get centroid in the 5 x 5 box */
				/* FIXME: Hogg check index logic here (2s) */
				for (oi=-1; oi<=1; oi++)
					for (oj=-1; oj<=1; oj++)
						three[oi + 1 + (oj + 1)*3] =
							simage[2*oi + xc[i] + (2 * oj + yc[i]) * onx];
				if (dcen3x3(three, &tmpxc, &tmpyc)) {
					xcen[imore + (*npeaks)] = 2.0 * tmpxc
						+ (float)(xc[i] + xmin - 2);
					ycen[imore + (*npeaks)] = 2.0 * tmpyc
						+ (float)(yc[i] + ymin - 2);
					assert(isfinite(xcen[imore + *npeaks]));
					assert(isfinite(ycen[imore + *npeaks]));
				} else {
					// don't add this peak.
					continue;
				}
			}
			imore++;
		}
		(*npeaks) += imore;

		l = lobj;
		nobj++;
	}

	FREEVEC(indx);
	FREEVEC(dobject);
	FREEVEC(oimage);
	FREEVEC(simage);
	FREEVEC(xc);
	FREEVEC(yc);

	return 1;

} /* end dallpeaks */
