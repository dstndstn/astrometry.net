/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, Dustin Lang, Keir Mierle and Sam Roweis.

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
#include <math.h>

#include "ppm.h"
#include "pnm.h"

int pnmutils_whitebalance(char* fn1, char* fn2,
						  double* Rgain, double* Ggain, double* Bgain) {
    FILE *fin1;
    FILE *fin2;
    int rows, cols, format;
    int rows2, cols2, format2;
	pixval maxval;
	pixval maxval2;
    int row;
    pixel * pixelrow1;
    pixel * pixelrow2;
	double redgain, bluegain, greengain;
	int nr, nb, ng;

    fin1 = pm_openr_seekable(fn1);
    fin2 = pm_openr_seekable(fn2);

    ppm_readppminit(fin1, &cols, &rows, &maxval, &format);
    ppm_readppminit(fin2, &cols2, &rows2, &maxval2, &format2);

	if (PNM_FORMAT_TYPE(format) != PPM_TYPE) {
		fprintf(stderr, "Input files must be PPM.\n");
		pm_close(fin1);
		pm_close(fin2);
		return -1;
	}
	if ((cols != cols2) || (rows != rows2) || (maxval != maxval2) || (format != format2)) {
		fprintf(stderr, "Images sizes, maxvals and formats must be the same.\n");
		pm_close(fin1);
		pm_close(fin2);
		return -1;
	}

    pixelrow1 = ppm_allocrow(cols);
    pixelrow2 = ppm_allocrow(cols);

	redgain = bluegain = greengain = 0.0;
	nr = nb = ng = 0;

    for (row = 0; row < rows; ++row) {
        int col;
        ppm_readppmrow(fin1, pixelrow1, cols, maxval, format);
        ppm_readppmrow(fin2, pixelrow2, cols, maxval, format);
        for (col = 0; col < cols; ++col) {
            const pixel p1 = pixelrow1[col];
            const pixel p2 = pixelrow2[col];
			double rf=1.0, gf=1.0, bf=1.0;

			if (p1.r && p2.r) {
				rf = (double)p1.r / (double)p2.r;
				nr++;
			}
			if (p1.g && p2.g) {
				gf = (double)p1.g / (double)p2.g;
				ng++;
			}
			if (p1.b && p2.b) {
				bf = (double)p1.b / (double)p2.b;
				nb++;
			}
			redgain += log(rf);
			greengain += log(gf);
			bluegain += log(bf);
		}
	}

	if (nr)
		redgain   = exp(redgain   / (double)nr);
	if (ng)
		greengain = exp(greengain / (double)ng);
	if (nb)
		bluegain  = exp(bluegain  / (double)nb);

    pnm_freerow(pixelrow1);
    pnm_freerow(pixelrow2);
    pm_close(fin1);
    pm_close(fin2);

	*Rgain = redgain;
	*Ggain = greengain;
	*Bgain = bluegain;

	return 0;
}
