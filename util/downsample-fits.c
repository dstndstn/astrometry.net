/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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
#include "qfits.h"
#include "log.h"
#include "errors.h"
#include "mathutil.h"
#include "fitsioutils.h"
#include "ioutils.h"

static const char* OPTIONS = "hvs:";

static void printHelp(char* progname) {
	printf("%s  [options]  <input-file> <output-file>\n"
		   "    use \"-\" to write to stdout.\n"
		   "      [-s <scale>]: downsample scale (default: 2): integer\n"
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
	bool tostdout = FALSE;
	qfitsloader load;
	qfitsdumper dump;
	qfits_header* hdr;
	float* img;
	int loglvl = LOG_MSG;
	int scale = 2;
	int window = 1024;
	int plane;
	int out_bitpix = -32;
	float* outimg;
	int outw, outh;
	int edge = EDGE_TRUNCATE;
	int ext = 0;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'v':
			loglvl++;
			break;
		case 's':
			scale = atoi(optarg);
			break;
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

	if (argc - optind != 2) {
		logerr("Need two arguments: input and output files.\n");
		printHelp(argv[0]);
		exit(-1);
	}

	infn = argv[optind];
	outfn = argv[optind+1];

	window = (int)round(window / (float)scale);

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
	load.map = 1;

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

	get_output_image_size(load.lx % scale, load.ly % scale,
						  scale, edge, &outw, &outh);
	outw += (load.lx / scale);
	outh += (load.ly / scale);

	hdr = qfits_header_default();
    fits_header_add_int(hdr, "BITPIX", out_bitpix, "bits per pixel");
	if (load.np > 1)
		fits_header_add_int(hdr, "NAXIS", 3, "number of axes");
	else
		fits_header_add_int(hdr, "NAXIS", 2, "number of axes");
    fits_header_add_int(hdr, "NAXIS1", outw, "image width");
    fits_header_add_int(hdr, "NAXIS2", outh, "image height");
	if (load.np > 1)
		fits_header_add_int(hdr, "NAXI3", load.np, "number of planes");
	if (qfits_header_dump(hdr, fout)) {
		ERROR("Failed to write FITS header to \"%s\"", outfn);
		exit(-1);
	}

	outimg = malloc((int)ceil(window/scale)*(int)ceil(window/scale) * sizeof(float));
			
	logmsg("Image is %i x %i x %i\n", load.lx, load.ly, load.np);
	logmsg("Output will be %i x %i x %i\n", outw, outh, load.np);
	logverb("Reading in blocks of %i x %i\n", window, window);
	for (plane=0; plane<load.np; plane++) {
		int bx, by;
		int nx, ny;
		load.pnum = plane;
		for (by=0; by<(int)ceil(load.ly / (float)window); by++) {
			for (bx=0; bx<(int)ceil(load.lx / (float)window); bx++) {
				int lox, loy, hix, hiy, outw, outh;
				nx = MIN(window, load.lx - bx*window);
				ny = MIN(window, load.ly - by*window);
				lox = 1 + bx*window;
				loy = 1 + by*window;
				hix = lox + nx;
				hiy = loy + ny;
				logverb("  reading %i,%i + %i,%i\n", lox, loy, nx, ny);
				if (qfits_loadpix_window(&load, lox, loy, hix, hiy)) {
					ERROR("Failed to load pixels.");
					exit(-1);
				}
				img = load.fbuf;

				average_image_f(img, nx, ny, scale, edge,
								&outw, &outh, outimg);

				dump.fbuf = outimg;
				dump.npix = outw * outh;
				if (qfits_pixdump(&dump)) {
					ERROR("Failed to write pixels.\n");
					exit(-1);
				}
			}
		}
	}
	qfitsloader_free_buffers(&load);
	free(outimg);

	if (!tostdout)
		fclose(fout);
	logverb("Done!\n");
	return 0;
}
