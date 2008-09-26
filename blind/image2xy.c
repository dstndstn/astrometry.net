/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Michael Blanton, Keir Mierle, David W. Hogg,
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

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/param.h>
#include <unistd.h>
#include <math.h>

#include "image2xy.h"
#include "ioutils.h"
#include "simplexy2.h"
#include "svn.h"
#include "dimage.h"
#include "errors.h"
#include "log.h"

static float* upconvert(unsigned char* u8,
                        int nx, int ny) {
    int i;
    float* f = malloc(nx * ny * sizeof(float));
    if (!f) {
        SYSERROR("Failed to allocate image array to upconvert u8 image to floating-point.");
        return NULL;
    }
    for (i=0; i<(nx*ny); i++)
        f[i] = u8[i];
    return f;
}

static void rebin(float** thedata,
                  int W, int H, int S,
                  int* newW, int* newH) {
    float sigma = S;
    int i, j, I, J;

    *newW = (W + S-1) / S;
    *newH = (H + S-1) / S;

    // Gaussian smooth in-place.
    dsmooth2(*thedata, W, H, sigma, *thedata);

    // Average SxS blocks, placing the result in the bottom (newW * newH) first pixels.
    for (j=0; j<*newH; j++) {
        for (i=0; i<*newW; i++) {
            float sum = 0.0;
            int N = 0;
            for (J=0; J<S; J++) {
                if (j*S + J >= H)
                    break;
                for (I=0; I<S; I++) {
                    if (i*S + I >= W)
                        break;
                    sum += (*thedata)[(j*S + J)*W + (i*S + I)];
                    N++;
                }
            }
            (*thedata)[j * (*newW) + i] = sum / (float)N;
        }
    }

    // save some memory...
    //*thedata = realloc(*thedata, (*newW) * (*newH) * sizeof(float));
}

int image2xy_image2(simplexy_t* s,
                    int downsample,
                    int downsample_as_required) {
	int fullW=-1, fullH=-1;
	int newW, newH;
	bool did_downsample = FALSE;
	bool free_fimage = FALSE;
	// the factor by which to downsample.
	int S = downsample ? downsample : 1;
	int jj;
    bool tryagain;
    int rtn = -1;

	fullW = s->nx;
	fullH = s->ny;
	if (downsample) {
		logmsg("Downsampling by %i...\n", S);
        if (!s->image) {
            s->image = upconvert(s->image_u8, s->nx, s->ny);
            free_fimage = TRUE;
        }
		if (!s->image)
			goto bailout;
		rebin(&s->image, s->nx, s->ny, S, &newW, &newH);
		s->nx = newW;
		s->ny = newH;
		did_downsample = TRUE;
	}

	do {
		simplexy2(s);

		tryagain = FALSE;
		if (s->npeaks == 0 &&
			downsample_as_required) {
			logmsg("Downsampling by 2...\n");
			if (s->image_u8) {
				s->image = upconvert(s->image_u8, s->nx, s->ny);
				if (!s->image)
					goto bailout;
				free_fimage = TRUE;
				s->image_u8 = NULL;
			}
			rebin(&s->image, s->nx, s->ny, 2, &newW, &newH);
			s->nx = newW;
			s->ny = newH;
			S *= 2;
			tryagain = TRUE;
			downsample_as_required--;
			did_downsample = TRUE;
		}
	} while (tryagain);

	for (jj=0; jj<s->npeaks; jj++) {
		assert(isfinite((s->x)[jj]));
		assert(isfinite((s->y)[jj]));
		// if S=1, this just shifts the origin to (1,1); the FITS
		// standard says the center of the lower-left pixel is (1,1).
		(s->x)[jj] = ((s->x)[jj] + 0.5) * (double)S + 0.5;
		(s->y)[jj] = ((s->y)[jj] + 0.5) * (double)S + 0.5;
	}

	dselip_cleanup();
    rtn = 0;
 bailout:
	if (free_fimage) {
		free(s->image);
        s->image = NULL;
    }
	return rtn;
}

int image2xy_image(uint8_t* u8image, float* fimage,
				   int W, int H,
				   int downsample, int downsample_as_required,
				   double dpsf, double plim, double dlim, double saddle,
				   int maxper, int maxsize, int halfbox, int maxnpeaks,
				   float** x, float** y, float** flux, float** background,
				   int* npeaks, float* sigma) {
    simplexy_t s;
    int rtn;

    if (u8image)
        simplexy2_set_u8_defaults(&s);
    else
        simplexy2_set_defaults(&s);

    if (dpsf)
        s.dpsf = dpsf;
    if (plim)
        s.plim = plim;
    if (dlim)
        s.dlim = dlim;
    if (saddle)
        s.saddle = saddle;
    if (maxper)
        s.maxper = maxper;
    if (maxsize)
        s.maxsize = maxsize;
    if (halfbox)
        s.halfbox = halfbox;
    if (maxnpeaks)
        s.maxnpeaks = maxnpeaks;

    s.image = fimage;
    s.image_u8 = u8image;
    s.nx = W;
    s.ny = H;

    rtn = image2xy_image2(&s, downsample, downsample_as_required);

    *x = s.x;
    *y = s.y;
    *flux = s.flux;
    *background = s.background;
    *npeaks = s.npeaks;
    *sigma = s.sigma;

    return rtn;
}

