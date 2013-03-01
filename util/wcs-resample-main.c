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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include <assert.h>

#include "wcs-resample.h"
#include "sip_qfits.h"
#include "qfits.h"
#include "starutil.h"
#include "bl.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"

const char* OPTIONS = "hw:e:E:x:";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] <input-FITS-image> <output (target) WCS-file> <output-FITS-image>\n"
		   "   [-E <input image FITS extension>] (default: 0)\n"
		   "   [-w <input WCS file>] (default is to read WCS from input FITS image)\n"
		   "   [-e <input WCS FITS extension>] (default: 0)\n"
		   "   [-x <output WCS FITS extension>] (default: 0)\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* inwcsfn = NULL;
	char* outwcsfn = NULL;
    char* infitsfn = NULL;
    char* outfitsfn = NULL;
	int inwcsext = 0;
	int inimgext = 0;
	int outwcsext = 0;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
        case 'w':
            inwcsfn = optarg;
            break;
		case 'e':
			inwcsext = atoi(optarg);
			break;
		case 'E':
			inimgext = atoi(optarg);
			break;
		case 'x':
			outwcsext = atoi(optarg);
			break;
		}
	}

    log_init(LOG_MSG);
    fits_use_error_system();

	if (optind != argc - 3) {
		print_help(args[0]);
		exit(-1);
	}

    infitsfn  = args[optind+0];
    outwcsfn  = args[optind+1];
    outfitsfn = args[optind+2];

    if (!inwcsfn)
        inwcsfn = infitsfn;

	if (resample_wcs_files(infitsfn, inimgext, inwcsfn, inwcsext,
						   outwcsfn, outwcsext, outfitsfn, 0)) {
		ERROR("Failed to resample image");
		exit(-1);
	}
	return 0;
}
