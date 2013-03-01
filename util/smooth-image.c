/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2010 Dustin Lang.

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
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "an-bool.h"
#include "convolve-image.h"
#include "qfits.h"
#include "log.h"
#include "errors.h"
#include "mathutil.h"
#include "fitsioutils.h"
#include "ioutils.h"

static const char* OPTIONS = "hvn:e:";

static void printHelp(char* progname) {
	printf("%s  [options]  <sigma> <input-file> <output-file>\n"
		   "    use \"-\" to write to stdout.\n"
		   "      [-e <extension>]: read extension (default: 0)\n"
		   "      [-n <n-sigma>]: number of sigmas at which to truncate (default: 5)\n"
		   "      [-v]: verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	FILE* fout;
	anbool tostdout = FALSE;
	qfitsloader load;
	qfitsdumper dump;
	qfits_header* hdr;
	float* img;
	int loglvl = LOG_MSG;
	double sigma = 0.0;
	int plane;
	int out_bitpix = -32;
	float* outimg;
	float* tempimg;
	int W, H;
	int ext = 0;
	int npixout = 0;
	double nsigma = 5;
	float* kernel = NULL;
	int NK;
	int K0;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'v':
			loglvl++;
			break;
		case 'e':
			ext = atoi(optarg);
			break;
		case 'n':
			nsigma = atof(optarg);
        case '?':
        case 'h':
			printHelp(argv[0]);
            return 0;
        default:
            return -1;
        }

	log_init(loglvl);
	log_to(stderr);
	errors_log_to(stderr);
	fits_use_error_system();

	if (argc - optind != 3) {
		printHelp(argv[0]);
		exit(-1);
	}

	sigma = atof(argv[optind]);
	infn = argv[optind+1];
	outfn = argv[optind+2];

	if (streq(outfn, "-")) {
		tostdout = TRUE;
		fout = stdout;
	} else {
		fout = fopen(outfn, "wb");
		if (!fout) {
			SYSERROR("Failed to open output file \"%s\"", outfn);
			exit(-1);
		}
	}

	load.filename = infn;
	load.xtnum = ext;
	load.pnum = 0;
	load.ptype = PTYPE_FLOAT;
	load.map = 0;

	if (qfitsloader_init(&load)) {
		ERROR("Failed to read input file info: \"%s\"", infn);
		exit(-1);
	}

	if (tostdout)
		dump.filename = "STDOUT";
	else
		dump.filename = outfn;
	dump.ptype = PTYPE_FLOAT;
	dump.out_ptype = out_bitpix;

	W = load.lx;
	H = load.ly;

	//hdr = qfits_header_read(

	hdr = qfits_header_default();
    fits_header_add_int(hdr, "BITPIX", out_bitpix, "bits per pixel");
	if (load.np > 1)
		fits_header_add_int(hdr, "NAXIS", 3, "number of axes");
	else
		fits_header_add_int(hdr, "NAXIS", 2, "number of axes");
    fits_header_add_int(hdr, "NAXIS1", W, "image width");
    fits_header_add_int(hdr, "NAXIS2", H, "image height");
	if (load.np > 1)
		fits_header_add_int(hdr, "NAXI3", load.np, "number of planes");
	if (qfits_header_dump(hdr, fout)) {
		ERROR("Failed to write FITS header to \"%s\"", outfn);
		exit(-1);
	}
	qfits_header_destroy(hdr);
	// qfits pixel dumping works by re-opening the file and appending to it... ugh...
	if (!tostdout && fclose(fout)) {
		ERROR("Failed to pad or close output file");
		exit(-1);
	}

	kernel = convolve_get_gaussian_kernel_f(sigma, nsigma, &K0, &NK);

	outimg = malloc(W * H * sizeof(float));
	tempimg = malloc(W * H * sizeof(float));

	logmsg("Image is %i x %i x %i\n", W, H, load.np);
	for (plane=0; plane<load.np; plane++) {
		load.pnum = plane;
		if (qfits_loadpix(&load)) {
			ERROR("Failed to load pixels");
			exit(-1);
		}
		img = load.fbuf;

		convolve_separable_f(img, W, H, kernel, K0, NK, outimg, tempimg);

		qfitsloader_free_buffers(&load);
		dump.fbuf = outimg;
		dump.npix = W * H;
		if (qfits_pixdump(&dump)) {
			ERROR("Failed to write pixels.\n");
			exit(-1);
		}
		npixout += dump.npix;
	}
	free(outimg);
	free(tempimg);
	free(kernel);


	if (tostdout) {
		// pad.
		int N;
		char pad[2880];
		N = (npixout * (abs(out_bitpix) / 8)) % 2880;
		memset(pad, 0, 2880);
		fwrite(pad, 1, N, fout);
	} else
		if (fits_pad_file_name(outfn)) {
			ERROR("Failed to pad output file");
			exit(-1);
		}

	logverb("Done!\n");
	return 0;
}
