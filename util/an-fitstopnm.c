/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "an-bool.h"
#include "qfits.h"
#include "permutedsort.h"
#include "qfits_error.h"

static const char* OPTIONS = "hi:o:Oe:p:m:I";

static void printHelp(char* progname) {
	printf("%s    -i <input-file>\n"
		   "      [-o <output-file>]       (default stdout)\n"
		   "      [-e <extension-number>]  FITS extension (default 0)\n"
		   "      [-p <plane-number>]      Image plane number (default 0)\n"
		   "      [-m <margin>]            Number of pixels to avoid at the image edges (default 0)\n"
		   "      [-O]: do ordinal transform (default: map 25-95 percentile)\n"
           "      [-I]: invert black-on-white image\n"
		   "\n", progname);
}

#define min(a,b) (((a)<(b))?(a):(b))

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	FILE* fout = NULL;
	char* progname = argv[0];
	int ordinal = 0;
	int ext = 0;
	int plane = 0;
	int margin = 0;
	qfitsloader ldr;
	float* img;
	int nx, ny;
    bool invert = FALSE;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'I':
            invert = TRUE;
            break;
        case 'i':
			infn = optarg;
			break;
        case 'o':
			outfn = optarg;
			break;
		case 'O':
			ordinal = 1;
			break;
		case 'e':
			ext = atoi(optarg);
			break;
		case 'p':
			plane = atoi(optarg);
			break;
		case 'm':
			margin = atoi(optarg);
			break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!infn) {
		printHelp(progname);
		exit(-1);
	}

	if (outfn) {
		fout = fopen(outfn, "wb");
		if (!fout) {
			fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
			exit(-1);
		}
	} else {
		fout = stdout;
	}

	// turn on QFITS error reporting.
	qfits_err_statset(1);

	ldr.filename = infn;
	ldr.xtnum = ext;
	ldr.pnum = plane;
	ldr.ptype = PTYPE_FLOAT;
	ldr.map = 1;

	if (qfitsloader_init(&ldr)) {
		fprintf(stderr, "Failed to read input file info: %s\n", infn);
		exit(-1);
	}

	if (qfits_loadpix(&ldr)) {
		fprintf(stderr, "Failed to load pixels.\n");
		exit(-1);
	}

	img = ldr.fbuf;
	nx = ldr.lx;
	ny = ldr.ly;

	if (ordinal) {
		int* perm;
		unsigned char* outimg;
		int i;
		int np = nx*ny;

		perm = permuted_sort(img, sizeof(float), compare_floats, NULL, np);

		qfitsloader_free_buffers(&ldr);

		outimg = malloc(np);
		for (i=0; i<(np); i++)
			outimg[perm[i]] = (unsigned char)(255.0 * (double)i / (double)(np+1));
		free(perm);

		fprintf(fout, "P5 %i %i 255\n", nx, ny);
		if (fwrite(outimg, 1, np, fout) != np) {
			fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
			exit(-1);
		}
		free(outimg);

	} else {
		// the maximum number of pixels to sample
		int NPIX = 10000;
		int n, np;
		int x, y;
		int i, j;
		float* pix;
		float maxval, minval, scale;

		fprintf(stderr, "Computing image percentiles...\n");
		n = (nx - 2*margin) * (ny - 2*margin);
		np = min(n, NPIX);
		pix = malloc(np * sizeof(float));
		if (n < NPIX) {
			i=0;
			for (y=margin; y<(ny-margin); y++)
				for (x=margin; x<(nx-margin); x++) {
					pix[i] = img[y*nx + x];
					i++;
				}
		} else {
			for (i=0; i<NPIX; i++) {
				x = margin + (nx - 2*margin) * ( (double)random() / (((double)RAND_MAX)+1.0) );
				y = margin + (ny - 2*margin) * ( (double)random() / (((double)RAND_MAX)+1.0) );
				pix[i] = img[y*nx + x];
			}
		}
		// I'm lazy: just sort it, even though we just need two percentiles.
		qsort(pix, np, sizeof(float), compare_floats);

        if (!invert) {
            // 25th percentile.
            j = (int)((0.25) * np) - 1;
            minval = pix[j];
            // 95th percentile.
            j = (int)((0.95) * np) - 1;
            maxval = pix[j];
        } else {
            j = (int)((0.75) * np) - 1;
            minval = pix[j];
            j = (int)((0.05) * np) - 1;
            maxval = pix[j];
        }

		scale = (255.0 / (maxval - minval));

		free(pix);

		fprintf(fout, "P5 %i %i 255\n", nx, ny);
		for (i=0; i<(nx*ny); i++) {
			float f = (img[i] - minval) * scale;
			if (f < 0) f = 0.0;
			if (f >= 256.0) f = 255.0;
            // FIXME - could count how many pixels are under/over-saturated
            // and readjust the scaling if our estimate of the percentiles
            // are very far off.
			fputc((int)f, fout);
		}

		qfitsloader_free_buffers(&ldr);
	}


	if (outfn)
		fclose(fout);

	return 0;
}
