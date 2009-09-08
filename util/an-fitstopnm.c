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

#include <sys/param.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "an-bool.h"
#include "qfits.h"
#include "permutedsort.h"
#include "qfits_error.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"

static const char* OPTIONS = "hi:o:Oe:p:m:IX:N:xnrsvM";

static void printHelp(char* progname) {
	printf("%s    -i <input-file>\n"
		   "      [-o <output-file>]       (default stdout)\n"
		   "      [-e <extension-number>]  FITS extension (default 0)\n"
		   "      [-p <plane-number>]      Image plane number (default 0)\n"
		   "      [-m <margin>]            Number of pixels to avoid at the image edges (default 0)\n"
		   "      [-O]: do ordinal transform (default: map 25-95 percentile)\n"
           "      [-I]: invert black-on-white image\n"
		   "      [-X <max>]: set the input value that will become white\n"
		   "      [-N <min>]: set the input value that will become black\n"
		   "      [-x]: set max to the observed maximum value\n"
		   "      [-n]: set min to the observed minimum value\n"
		   "      [-r]: same as -x -n: set min and max to observed data range.\n"
		   "      [-s]: write 16-bit output\n"
		   "      [-v]: verbose\n"
		   "      [-M]: compute & print median value\n"
		   "\n", progname);
}


static void sample_percentiles(const float* img, int nx, int ny, int margin,
							   int NPIX, float lop, float hip,
							   float* lo, float* hi) {
	// the maximum number of pixels to sample
	int n, np;
	int x, y;
	int i;
	float* pix;

	fprintf(stderr, "Computing image percentiles...\n");
	n = (nx - 2*margin) * (ny - 2*margin);
	np = MIN(n, NPIX);
	pix = malloc(np * sizeof(float));
	if (n < NPIX) {
		i=0;
		for (y=margin; y<(ny-margin); y++)
			for (x=margin; x<(nx-margin); x++) {
				pix[i] = img[y*nx + x];
				i++;
			}
		assert(i == np);
	} else {
		for (i=0; i<np; i++) {
			x = margin + (nx - 2*margin) * ( (double)random() / (((double)RAND_MAX)+1.0) );
			y = margin + (ny - 2*margin) * ( (double)random() / (((double)RAND_MAX)+1.0) );
			pix[i] = img[y*nx + x];
		}
	}

	qsort(pix, np, sizeof(float), compare_floats_asc);

	if (lo) {
		i = MIN(np-1, MAX(0, (int)(lop * np)));
		*lo = pix[i];
	}
	if (hi) {
		i = MIN(np-1, MAX(0, (int)(hip * np)));
		*hi = pix[i];
	}
	free(pix);
}
							   


extern char *optarg;
extern int optind, opterr, optopt;

#define NBUF 1024

int main(int argc, char *argv[]) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	FILE* fout = NULL;
	char* progname = argv[0];
	bool ordinal = FALSE;
	int ext = 0;
	int plane = 0;
	int margin = 0;
	qfitsloader ldr;
	float* img;
	int nx, ny;
    bool invert = FALSE;

	bool minval_set = FALSE;
	bool maxval_set = FALSE;
	float maxval, minval;
	bool find_min = FALSE;
	bool find_max = FALSE;

	bool sixteenbit = FALSE;
	int maxpix;
	int loglvl = LOG_MSG;
	bool median = FALSE;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'v':
			loglvl++;
			break;
		case 's':
			sixteenbit = TRUE;
			break;
		case 'X':
			maxval = atof(optarg);
			maxval_set = TRUE;
			break;
		case 'N':
			minval = atof(optarg);
			minval_set = TRUE;
			break;
		case 'x':
			find_max = TRUE;
			break;
		case 'n':
			find_min = TRUE;
			break;
		case 'r':
			find_min = TRUE;
			find_max = TRUE;
			break;
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
			ordinal = TRUE;
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
		case 'M':
			median = TRUE;
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

	log_init(loglvl);
	log_to(stderr);
	errors_log_to(stderr);
	fits_use_error_system();

	if (outfn) {
		fout = fopen(outfn, "wb");
		if (!fout) {
			SYSERROR("Failed to open output file \"%s\"", outfn);
			exit(-1);
		}
	} else {
		fout = stdout;
	}

	maxpix = (sixteenbit ? 65535 : 255);

	ldr.filename = infn;
	ldr.xtnum = ext;
	ldr.pnum = plane;
	ldr.ptype = PTYPE_FLOAT;
	ldr.map = 1;

	if (qfitsloader_init(&ldr)) {
		ERROR("Failed to read input file info: \"%s\"", infn);
		exit(-1);
	}

	logverb("Reading pixels...\n");
	if (qfits_loadpix(&ldr)) {
		ERROR("Failed to load pixels.");
		exit(-1);
	}

	img = ldr.fbuf;
	nx = ldr.lx;
	ny = ldr.ly;

	if (median) {
		int* perm = permuted_sort(img, sizeof(float), compare_floats_asc, NULL, nx*ny);
		logmsg("Median value: %g\n", img[perm[(nx*ny)/2]]);
		free(perm);
	}

	if (ordinal) {
		int* perm;
		unsigned char* outimg;
		int i;
		int np = nx*ny;

		logverb("Doing ordinal transform...\n");
		perm = permuted_sort(img, sizeof(float), compare_floats_asc, NULL, np);

		qfitsloader_free_buffers(&ldr);

		if (sixteenbit)
			outimg = malloc(np * sizeof(uint16_t));
		else
			outimg = malloc(np);

		if (invert) {
			for (i=0; i<np; i++)
				outimg[perm[i]] = (unsigned char)(maxpix * (double)(np-1 - i) / (double)(np));
		} else {
			for (i=0; i<np; i++)
				outimg[perm[i]] = (unsigned char)(maxpix * (double)i / (double)(np));
		}
		free(perm);

		logverb("Writing output...\n");
		fprintf(fout, "P5 %i %i %i\n", nx, ny, maxpix);
		if (sixteenbit)
			for (i=0; i<np; i++)
				outimg[i] = htons(outimg[i]);

		if (fwrite(outimg, sixteenbit ? 2 : 1, np, fout) != np) {
			fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
			exit(-1);
		}
		free(outimg);

	} else {
		int i, j;
		float scale;

		if (find_min) {
			minval = HUGE_VALF;
			for (i=0; i<(nx*ny); i++)
				minval = MIN(minval, img[i]);
			minval_set = TRUE;
			logverb("Minimum pixel value: %g\n", minval);
		}
		if (find_max) {
			maxval = -HUGE_VALF;
			for (i=0; i<(nx*ny); i++)
				maxval = MAX(maxval, img[i]);
			maxval_set = TRUE;
			logverb("Maximum pixel value: %g\n", maxval);
		}

		if (!(minval_set && maxval_set)) {
			float lop, hip;
			int NPIX = 10000;

			// percentiles.
			if (invert) {
				lop = 0.05;
				hip = 0.75;
			} else {
				lop = 0.25;
				hip = 0.95;
			}

			logverb("Computing image percentiles...\n");
			sample_percentiles(img, nx, ny, margin, NPIX, lop, hip,
							   (minval_set ? NULL : &minval),
							   (maxval_set ? NULL : &maxval));
		}

		if (invert) {
			scale = -((float)maxpix / (maxval - minval));
			minval = maxval;
		} else
			scale = ((float)maxpix / (maxval - minval));

		logverb("Mapping input pixel range [%f, %f]\n", minval, maxval);
		logverb("Writing output..\n");
		fprintf(fout, "P5 %i %i %i\n", nx, ny, maxpix);
		i = 0;
		while (i < (nx*ny)) {
			int n;
			n = MIN(NBUF, nx*ny - i);
			if (sixteenbit) {
				uint16_t buf[NBUF];
				for (j=0; j<n; j++)
					buf[j] = htons(MIN(65535, MAX(0, round((img[i+j] - minval) * scale))));
				if (fwrite(buf, 2, n, fout) != n) {
					fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
					exit(-1);
				}
				
			} else {
				uint8_t buf[NBUF];
				for (j=0; j<n; j++)
					buf[j] = MIN(255, MAX(0, round((img[i+j] - minval) * scale)));
				if (fwrite(buf, 1, n, fout) != n) {
					fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
					exit(-1);
				}
			}
			i += n;
		}
		qfitsloader_free_buffers(&ldr);
	}

	if (outfn)
		fclose(fout);
	logverb("Done!\n");
	return 0;
}
