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

#ifndef IMAGE2XY_H
#define IMAGE2XY_H

#include <stdint.h>

#include "an-bool.h"

/**
 Reads an input FITS image (possibly multi-HDU), runs simplexy on each
 image and places the results in an output file containing a FITS BINTABLE.
 */
int image2xy(const char* infn, const char* outfn,
             bool do_u8, int downsample,
             int downsample_as_required);


/*
 */
int image2xy_image(uint8_t* u8image, float* fimage,
				   int W, int H,
				   int downsample, int downsample_as_required,
				   double dpsf, double plim, double dlim, double saddle,
				   int maxper, int maxsize, int halfbox, int maxnpeaks,
				   float** x, float** y, float** flux, float** background,
				   int* npeaks, float* sigma);

#endif
