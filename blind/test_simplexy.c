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
#include <math.h>

#include "cutest.h"

extern int initial_max_groups;

int simplexy(float *image, int nx, int ny, float dpsf, float plim, float dlim,
		float saddle, int maxper, int maxnpeaks, int maxsize, int halfbox,
		float *sigma, float *x, float *y, float *flux, int *npeaks, int
		verbose);

static void run(float *im, int w, int h, float *x, float *y, float *flux, int *n) {
	float sigma;
	double dpsf = 1.0;
	double plim = 8.0;
	double dlim = dpsf;
	double saddle = 5.0;
	double maxper = 1000;
	double maxsize = 1000;
	double halfbox = 7;
	double maxnpeaks = 100;
	simplexy(im, w, h,
			dpsf, plim, dlim, saddle, maxper, maxnpeaks,
			maxsize, halfbox, &sigma, x, y, flux, n, 1);
}

void test_one_peak(CuTest* tc) {
	float x[100];
	float y[100];
	float flux[100];
	int n;
	float test_data[] = {
		7,  7,  6,  5,  7,  7,  5,  5,  6,  7,  6,  7,  6,  5,  5,  5,  5,
        6,  7,  8,  5,  6,  6,  5,  5,  6,  6,  6,  7,  7,  7,  6,  5,  5,
        5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  8,  7,  6,  4,
        6,  4,  4,  7,  6,  7,  8,  8,  6,  6,  7,  4,  5,  7,  7,  6,  5,
        8,  8,  8, 10,  4,  9,  6,  8,  4,  3,  5,  6,  6,  6,  5,  4,  4,
        9,  9,  9,  7, 10,  5,  8,  4,  4, 11,  4,  7,  6,  6,  6,  6,  5,
        8,  8,  7,  7, 10,  4,  5,  7,  7,  6, 14,  8,  5,  5,  7,  8,  6,
        7,  6,  7,  5,  8,  9,  8,  8,  2,  7,  4,  7,  5,  5,  8,  8,  6,
        6,  7,  9, 10,  8,  7,  6,  5, 17, 75, 41,  6,  5,  6,  7,  6,  4,
        6,  8,  9,  8,  3,  6,  7,  7, 21, 70, 43,  6,  6,  7,  7,  5,  4,
        8,  8,  6, 10,  6,  8,  6,  6,  6, 10, 10,  6,  7,  7,  6,  5,  6,
       11, 10,  7,  6, 11,  4, 10,  5,  8,  2,  6,  7,  8,  7,  5,  6,  8,
       10, 11,  9,  9,  7,  7,  9,  9,  7,  6,  7,  5, 10,  9,  6,  6,  7,
        9,  8,  7,  7,  7,  7,  7,  7,  8,  8,  7,  6,  9,  7,  5,  6,  7,
        9,  8,  9,  7,  8,  8,  6,  7,  9,  9,  7,  7,  8,  6,  4,  6,  7,
       11,  9, 10,  9, 10,  9,  8,  8,  9,  9,  8,  7,  7,  6,  5,  6,  7,
       11,  8,  7,  9,  8,  8,  9,  9,  8,  9, 10,  6,  7,  8,  7,  6,  7,
       10,  8,  6,  8,  6,  7,  9,  9,  7,  8, 10,  6,  7,  9,  9,  7,  6
	};
	run(test_data, 17, 18, x,y,flux,&n);
	CuAssertIntEquals(tc, 1, n);
	CuAssert(tc, "single peak is in the middle", fabs(x[0]-8.3) < 0.1);
	CuAssert(tc, "single peak is in the middle", fabs(y[0]-8.2) < 0.1);
}
