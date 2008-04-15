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

/*
*/

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <sys/param.h>

#include "pm.h"
#include "ppm.h"
#include "pnm.h"

static const char* OPTIONS = "hr:g:b:";

static void printHelp(char* progname) {
	fprintf(stderr, "usage: %s\n"
			"    [-r <red-gain>  ]\n"
			"    [-g <green-gain>]\n"
			"    [-b <blue-gain> ]\n"
			"    <input-image>\n"
			"Writes PPM on stdout.\n"
			"\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int
main(int argc, char** args) {
	char* progname = args[0];
	int argchar;
    FILE *fin;
	char* fn;
    int rows, cols, format;
	pixval rmax, gmax, bmax, maxval;
    int row;
    pixel * pixelrow;
	FILE *fout;
	off_t imgstart;
	pixval *rmap, *bmap, *gmap;
	int i;
	double redgain, bluegain, greengain;

	redgain = bluegain = greengain = 0.0;

    pnm_init(&argc, args);

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'h':
			printHelp(progname);
			exit(0);
		case 'r':
			redgain = atof(optarg);
			break;
		case 'g':
			greengain = atof(optarg);
			break;
		case 'b':
			bluegain = atof(optarg);
			break;
		}

	if (optind == argc) {
		printHelp(progname);
		exit(-1);
	}
	fn = args[optind];

    fin = pm_openr_seekable(fn);

    ppm_readppminit(fin, &cols, &rows, &maxval, &format);

	if (PNM_FORMAT_TYPE(format) != PPM_TYPE) {
		fprintf(stderr, "Input file must be PPM.\n");
		exit(-1);
	}

    pixelrow = ppm_allocrow(cols);

	rmap = malloc(maxval * sizeof(pixval));
	gmap = malloc(maxval * sizeof(pixval));
	bmap = malloc(maxval * sizeof(pixval));
	if (!rmap || !gmap || !bmap) {
		fprintf(stderr, "Failed to allocate color maps.\n");
		exit(-1);
	}

	imgstart = ftello(fin);

	rmax = gmax = bmax = 0;
    for (row = 0; row < rows; ++row) {
        int col;
        ppm_readppmrow(fin, pixelrow, cols, maxval, format);
        for (col = 0; col < cols; ++col) {
            pixel const p = pixelrow[col];
			if (p.r > rmax) rmax = p.r;
			if (p.g > gmax) gmax = p.g;
			if (p.b > bmax) bmax = p.b;
		}
	}

	fout = stdout;
	ppm_writeppminit(fout, cols, rows, maxval, 0);

	fseeko(fin, imgstart, SEEK_SET);

	for (i=0; i<maxval; i++) {
		if (redgain == 0.0)
			rmap[i] = (pixval)rint((double)i * maxval / (double)rmax);
		else
			rmap[i] = (pixval)MIN(maxval, rint((double)i * redgain));

		if (greengain == 0.0)
			gmap[i] = (pixval)rint((double)i * maxval / (double)gmax);
		else
			gmap[i] = (pixval)MIN(maxval, rint((double)i * greengain));
			
		if (bluegain == 0.0)
			bmap[i] = (pixval)rint((double)i * maxval / (double)bmax);
		else
			bmap[i] = (pixval)MIN(maxval, rint((double)i * bluegain));
	}

    for (row = 0; row < rows; ++row) {
        int col;
        ppm_readppmrow(fin, pixelrow, cols, maxval, format);
        for (col = 0; col < cols; ++col) {
            pixel p = pixelrow[col];
			p.r = rmap[p.r];
			p.g = gmap[p.g];
			p.b = bmap[p.b];
			pixelrow[col] = p;
		}
		ppm_writeppmrow(fout, pixelrow, cols, maxval, 0);
	}
	
    pnm_freerow(pixelrow);

    pm_close(fin);
    return 0;
}
