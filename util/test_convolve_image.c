/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.

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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"
#include "convolve-image.h"
#include "fitsioutils.h"

void test_conv_1(CuTest* tc) {
	int W = 13;
	int H = 11;
	int i;
	float* img = malloc(W * H * sizeof(float));
	float* kernel;
	float* cimg;
	int K0, NK;
	double sig = 1.0;

	for (i=0; i<(W*H); i++)
		img[i] = 0;
	img[(H/2)*W + (W/2)] = 1.;

	kernel = convolve_get_gaussian_kernel_f(sig, 5., &K0, &NK);
	cimg = convolve_separable_f(img, W, H, kernel, K0, NK, NULL, NULL);

	fits_write_float_image(cimg, W, H, "test-conv.fits");

}


