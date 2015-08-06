/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

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
#include "simplexy.h"
#include "dimage.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"

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

	get_output_image_size(W, H, S, EDGE_AVERAGE, newW, newH);

    // Gaussian smooth in-place.
    dsmooth2(*thedata, W, H, sigma, *thedata);

    // Average SxS blocks, placing the result in the bottom (newW * newH) first pixels.
	if (!average_image_f(*thedata, W, H, S, EDGE_AVERAGE, newW, newH, *thedata)) {
		ERROR("Averaging the image failed.");
		return;
	}
}

int image2xy_run(simplexy_t* s,
				 int downsample, int downsample_as_required) {
	int newW, newH;
	anbool free_fimage = FALSE;
	// the factor by which to downsample.
	int S = downsample ? downsample : 1;
	int jj;
    anbool tryagain;
    int rtn = -1;

	if (downsample && downsample > 1) {
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
	}

	do {
		simplexy_run(s);

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
		}
	} while (tryagain);

	for (jj=0; jj<s->npeaks; jj++) {
		assert(isfinite((s->x)[jj]));
		assert(isfinite((s->y)[jj]));
		// shift the origin to the FITS standard: 
		// center of the lower-left pixel is (1,1).
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

