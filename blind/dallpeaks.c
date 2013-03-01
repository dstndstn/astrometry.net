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
#include "log.h"
#include "mathutil.h"

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

static int max_gaussian(float* image, int W, int H, float sigma,
						int x0, int y0, float *p_x, float *p_y) {
	float x, y;
	int step;
	float stepsize = 0.1;
	// inverse-variance
	float iv = 1.0 / (sigma*sigma);
	// half inverse-variance
	float hiv = 1.0 / (2.0 * sigma*sigma);
	// clip at six sigma:
	int Nsigma = 6;
	// Nsigma=6: G=1.523e-08

	x = x0;
	y = y0;

	for (;;) {
		float xdir=0, ydir=0;
		anbool xflipped = FALSE, yflipped = FALSE;

		for (step=0; step<100; step++) {
			float dx, dy;
			float Gx, Gy;
			float V;
			int i,j;
			int ilo,ihi,jlo,jhi;

			debug("Stepsize %g, step %i\n", stepsize, step);

			V = Gx = Gy = 0;
			// compute gradients.
			ilo = MAX(0, floor(x-Nsigma*sigma));
			ihi = MIN(W-1, ceil(x+Nsigma*sigma));
			jlo = MAX(0, floor(y-Nsigma*sigma));
			jhi = MIN(H-1, ceil(y+Nsigma*sigma));
			//for (j=0; j<H; j++) {
			//for (i=0; i<W; i++) {
			for (j=jlo; j<=jhi; j++) {
				for (i=ilo; i<=ihi; i++) {
					float G;
					dx = i - x;
					dy = j - y;
					// ~ Gaussian
					G = exp(-(dx*dx + dy*dy) * hiv);
					// other common factors
					G *= (image[j*W + i] * iv);
					// Gradients
					Gx += (G * -dx);
					Gy += (G * -dy);
					V += G;
				}
			}
			debug("x,y = (%g,%g), V=%g, Gx=%g, Gy=%g\n", x, y, V, Gx, Gy);

			dx = SIGN(-Gx);
			dy = SIGN(-Gy);

			if (step == 0) {
				if (xdir == 0 && ydir == 0)
					break;
				xdir = dx;
				ydir = dy;
			}
			if (!xflipped && dx == xdir)
				x += dx * stepsize;
			else
				xflipped = TRUE;
			if (!yflipped && dy == ydir)
				y += dy * stepsize;
			else
				yflipped = TRUE;
			if (xflipped && yflipped)
				break;
		}
		if (stepsize <= 0.002)
			break;
		stepsize /= 10.0;
	}
	if (p_x)
		*p_x = x;
	if (p_y)
		*p_y = y;
	return 0;
}


#define IMGTYPE float
#define SUFFIX
#include "dallpeaks.inc"
#undef SUFFIX
#undef IMGTYPE

#define IMGTYPE uint8_t
#define SUFFIX _u8
#include "dallpeaks.inc"
#undef IMGTYPE
#undef SUFFIX

#define IMGTYPE int16_t
#define SUFFIX _i16
#include "dallpeaks.inc"
#undef IMGTYPE
#undef SUFFIX
