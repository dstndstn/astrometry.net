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

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/param.h>
#include <errno.h>
#include <unistd.h>

#include "image2xy.h"
#include "log.h"
#include "errors.h"
#include "ioutils.h"

static const char* OPTIONS = "hOo:8Hd:v";

static void printHelp() {
	fprintf(stderr,
			"Usage: image2xy [options] fitsname.fits \n"
			"\n"
			"Read a FITS file, find objects, and write out \n"
			"X, Y, FLUX to   fitsname.xy.fits .\n"
			"\n"
			"   [-O]  overwrite existing output file.\n"
            "   [-8]  don't use optimization for byte (u8) images.\n"
            "   [-H]  downsample by a factor of 2 before running simplexy.\n"
            "   [-d <downsample-factor>]  downsample by an integer factor before running simplexy.\n"
			"   [-o <output-filename>]  write XYlist to given filename.\n"
            "   [-v] verbose - repeat for more and more verboseness\n"
			"\n"
			"   image2xy 'file.fits[1]'   - process first extension.\n"
			"   image2xy 'file.fits[2]'   - process second extension \n"
			"   image2xy file.fits+2      - same as above \n"
			"\n");
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;
	char* outfn = NULL;
	char* infn;
	int overwrite = 0;
    int loglvl = LOG_MSG;
    bool do_u8 = TRUE;
    int downsample = 0;
    

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'H':
            downsample = 2;
            break;
        case 'd':
            downsample = atoi(optarg);
            break;
        case '8':
            do_u8 = FALSE;
            break;
        case 'v':
            loglvl++;
            break;
		case 'O':
			overwrite = 1;
			break;
		case 'o':
			outfn = optarg;
			break;
		case '?':
		case 'h':
			printHelp();
			exit(0);
		}

	if (optind != argc - 1) {
		printHelp();
		exit(-1);
	}

	infn = argv[optind];

    log_init(loglvl);
    logverb("infile=%s\n", infn);

	if (!outfn) {
		// Create xylist filename (by trimming '.fits')
		asprintf(&outfn, "%.*s.xy.fits", (int)(strlen(infn)-5), infn);
        logverb("outfile=%s\n", outfn);
	}

	if (overwrite && file_exists(outfn)) {
        logverb("Deleting existing output file \"%s\"...\n", outfn);
        if (unlink(outfn)) {
            SYSERROR("Failed to delete existing output file \"%s\"", outfn);
            exit(-1);
        }
	}

    if (downsample)
        logverb("Downsampling by %i\n", downsample);

    if (image2xy(infn, outfn, do_u8, downsample, 0)) {
        ERROR("image2xy failed.");
        exit(-1);
    }
	return 0;
}
