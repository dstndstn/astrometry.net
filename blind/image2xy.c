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

int image2xy_image(uint8_t* u8image, float* fimage,
				   int W, int H,
				   int downsample, int downsample_as_required,
				   double dpsf, double plim, double dlim, double saddle,
				   int maxper, int maxsize, int halfbox, int maxnpeaks,
				   float** x, float** y, float** flux, float** background,
				   int* npeaks, float* sigma) {
	int fullW=-1, fullH=-1;
	simplexy_t s;
	int newW, newH;
	bool did_downsample = FALSE;
	bool free_fimage = FALSE;
	// the factor by which to downsample.
	int S = downsample ? downsample : 1;
	int jj;
    bool tryagain;

	if (dpsf == 0)
		dpsf = IMAGE2XY_DEFAULT_DPSF;
	if (plim == 0)
		plim = IMAGE2XY_DEFAULT_PLIM;
	if (dlim == 0)
		dlim = IMAGE2XY_DEFAULT_DLIM;
	if (saddle == 0)
		saddle = IMAGE2XY_DEFAULT_SADDLE;
	if (maxper == 0)
		maxper = IMAGE2XY_DEFAULT_MAXPER;
	if (maxsize == 0)
		maxsize = IMAGE2XY_DEFAULT_MAXSIZE;
	if (halfbox == 0)
		halfbox = IMAGE2XY_DEFAULT_HALFBOX;
	if (maxnpeaks == 0)
		maxnpeaks = IMAGE2XY_DEFAULT_MAXNPEAKS;

	fullW = W;
	fullH = H;
	if (downsample) {
		logmsg("Downsampling by %i...\n", S);
        if (!fimage) {
            fimage = upconvert(u8image, W, H);
            free_fimage = TRUE;
        }
		if (!fimage)
			goto bailout;
		rebin(&fimage, W, H, S, &newW, &newH);
		W = newW;
		H = newH;
		did_downsample = TRUE;
	}

	do {
		memset(&s, 0, sizeof(simplexy_t));
		s.image = fimage;
		s.image_u8 = u8image;
		s.nx = W;
		s.ny = H;

		s.dpsf = dpsf;
		s.plim = plim;
		s.dlim = dlim;
		s.saddle = saddle;
		s.maxper = maxper;
		s.maxnpeaks = maxnpeaks;
		s.maxsize = maxsize;
		s.halfbox = halfbox;
        
		simplexy2(&s);

		*x = s.x;
		*y = s.y;
		*flux = s.flux;
		*background = s.background;
		*npeaks = s.npeaks;
		*sigma = s.sigma;

		tryagain = FALSE;
		if (s.npeaks == 0 &&
			downsample_as_required) {
			logmsg("Downsampling by 2...\n");
			if (u8image) {
				fimage = upconvert(u8image, W, H);
				if (!fimage)
					goto bailout;
				free_fimage = TRUE;
				u8image = NULL;
				s.image = fimage;
				s.image_u8 = u8image;
			}
			rebin(&fimage, W, H, 2, &newW, &newH);
			W = newW;
			H = newH;
			S *= 2;
			tryagain = TRUE;
			downsample_as_required--;
			did_downsample = TRUE;
		}
	} while (tryagain);

	for (jj=0; jj<s.npeaks; jj++) {
		assert(isfinite((*x)[jj]));
		assert(isfinite((*y)[jj]));
		// if S=1, this just shifts the origin to (1,1); the FITS
		// standard says the center of the lower-left pixel is (1,1).
		(*x)[jj] = ((*x)[jj] + 0.5) * (double)S + 0.5;
		(*y)[jj] = ((*y)[jj] + 0.5) * (double)S + 0.5;
	}

	if (free_fimage)
		free(fimage);

	dselip_cleanup();

	return 0;
 bailout:
	if (free_fimage)
		free(fimage);
	return -1;
}

