/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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
#include <string.h>
#include <math.h>

#include "qfits.h"
#include "fits-guess-scale.h"
#include "fitsioutils.h"
#include "log.h"

static char* OPTIONS = "hv";

static void printHelp(char* progname) {
	printf("%s  <FITS-file>\n\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
	char* progname = argv[0];
    int argchar;
	char* infn;

    sl* methods = NULL;
    dl* scales = NULL;
    int i;
    int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        case 'v':
            loglvl++;
            break;
        default:
            return -1;
        }

	if (optind != (argc - 1)) {
		printHelp(progname);
		exit(-1);
	}
	infn = argv[optind];

    log_init(loglvl);

    fits_use_error_system();

    if (fits_guess_scale(infn, &methods, &scales))
        exit(-1);

    for (i=0; i<sl_size(methods); i++) {
        printf("scale %s %g\n", sl_get(methods, i), dl_get(scales, i));
    }

    sl_free2(methods);
    dl_free(scales);

    return 0;
}

