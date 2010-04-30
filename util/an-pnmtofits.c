/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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

#include "an-bool.h"

//#include <netpbm/pam.h>
#include "pam.h"

#include "qfits_image.h"

#include "log.h"
#include "errors.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "bl.h"

static const char* OPTIONS = "hvqo:";

static void printHelp(char* progname) {
	printf("%s    [options]  [<input-file>, default stdin]\n"
		   "      or         [<input-file> <output-file>]\n"
		   "      [-o <output-file>]       (default stdout)\n"
		   "      [-v]: verbose\n"
		   "      [-q]: quiet\n"
		   "\n", progname);
}

	

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	struct pam img;
	tuple * tuplerow;
	unsigned int row;
	int bits;
	FILE* fid = stdin;
	FILE* fout = stdout;
	int loglvl = LOG_MSG;
	char* progname = args[0];
	int bzero = 0;
	int outformat;
	qfits_header* hdr;
	unsigned int plane;
	off_t datastart;
	bool onepass = FALSE;
	bl* pixcache = NULL;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case '?':
		case 'h':
			printHelp(progname);
			exit(0);
		case 'v':
			loglvl++;
			break;
		case 'q':
			loglvl--;
			break;
		case 'o':
			outfn = optarg;
			break;
		}

	log_init(loglvl);
	log_to(stderr);
	fits_use_error_system();

	if (optind == argc) {
		// ok, stdin to stdout.
	} else if (optind == argc-1) {
		infn = args[optind];
	} else if (optind == argc-2) {
		infn = args[optind];
		outfn = args[optind+1];
	} else {
		printHelp(progname);
		exit(-1);
	}

	if (infn && !streq(infn, "-")) {
		fid = fopen(infn, "rb");
		if (!fid) {
			SYSERROR("Failed to open input file %s", infn);
			exit(-1);
		}
	}
	if (outfn) {
		fout = fopen(outfn, "wb");
		if (!fid) {
			SYSERROR("Failed to open output file %s", outfn);
			exit(-1);
		}
	} else
		outfn = "stdout";

	pm_init(args[0], 0);
	pnm_readpaminit(fid, &img, PAM_STRUCT_SIZE(tuple_type));
	logmsg("Read file %s: %i x %i pixels x %i color(s); maxval %i\n",
		   infn ? infn : "stdin", img.width, img.height, img.depth, (int)img.maxval);

	tuplerow = pnm_allocpamrow(&img);

	bits = pm_maxvaltobits(img.maxval); 
	bits = (bits <= 8) ? 8 : 16;
	if (bits == 8)
		outformat = BPP_8_UNSIGNED;
	else {
		outformat = BPP_16_SIGNED;
		if (img.maxval >= INT16_MAX)
			bzero = 0x8000;
	}
	logmsg("Using %i-bit output\n", bits);

	hdr = fits_get_header_for_image3(img.width, img.height, outformat, img.depth, NULL);
	if (bzero)
		fits_header_add_int(hdr, "BZERO", bzero, "Amount that has been subtracted from pixel values");
	if (qfits_header_dump(hdr, fout)) {
		ERROR("Failed to write FITS header to file %s", outfn);
		exit(-1);
	}

	datastart = ftello(fid);
	// Figure out if we can seek backward in this input file...
	if ((fid == stdin) ||
		(fseeko(fid, 0, SEEK_SET) ||
		 fseeko(fid, datastart, SEEK_SET)))
		// Nope!
		onepass = TRUE;
	logmsg("Reading in one pass\n");
	if (onepass)
		pixcache = bl_new(16384, bits/8);

	for (plane=0; plane<img.depth; plane++) {
		if (plane > 0) {
			if (fseeko(fid, datastart, SEEK_SET)) {
				SYSERROR("Failed to seek back to start of image data");
				exit(-1);
			}
		}
		for (row = 0; row<img.height; row++) {
			unsigned int column;
			pnm_readpamrow(&img, tuplerow);
			for (column = 0; column<img.width; column++) {
				int rtn;
				//grand_total += tuplerow[column][plane];
				if (outformat == BPP_8_UNSIGNED)
					rtn = fits_write_data_B(fout, tuplerow[column][plane]-bzero);
				else
					rtn = fits_write_data_I(fout, tuplerow[column][plane]-bzero);
				if (rtn) {
					ERROR("Failed to write FITS pixel");
					exit(-1);
				}
			}
			if (onepass && img.depth > 1) {
				for (column = 0; column<img.width; column++) {
					for (plane=1; plane<img.depth; plane++) {
						if (outformat == BPP_8_UNSIGNED) {
							uint8_t pix = tuplerow[column][plane];
							bl_append(pixcache, &pix);
						} else {
							int16_t pix = tuplerow[column][plane] - bzero;
							bl_append(pixcache, &pix);
						}
					}
				}
			}
		}
	}
	pnm_freepamrow(tuplerow);

	if (pixcache) {
		int i, j;
		int step = (img.depth - 1);
		logverb("Writing %i queued pixels\n", bl_size(pixcache));
		for (plane=1; plane<img.depth; plane++) {
			j = (plane - 1);
			for (i=0; i<(img.width * img.height); i++) {
				int rtn;
				if (outformat == BPP_8_UNSIGNED) {
					uint8_t* pix = bl_access(pixcache, j);
					rtn = fits_write_data_B(fout, *pix);
				} else {
					int16_t* pix = bl_access(pixcache, j);
					rtn = fits_write_data_I(fout, *pix);
				}
				if (rtn) {
					ERROR("Failed to write FITS pixel");
					exit(-1);
				}
				j += step;
			}
		}
		bl_free(pixcache);
	}

	if (fid != stdin)
		fclose(fid);

	if (fout != stdout)
		if (fclose(fout)) {
			SYSERROR("Failed to close output file %s", outfn);
			exit(-1);
		}

	return 0;
}
