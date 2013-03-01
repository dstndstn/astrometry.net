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
#include <string.h>
#include <assert.h>

#include "qfits.h"
#include "an-bool.h"
#include "resort-xylist.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"

const char* OPTIONS = "hdf:b:v";

static void printHelp(char* progname) {
    printf("Usage:   %s  <input> <output>\n"
		   "      -f <flux-column-name>  (default: FLUX) \n"
		   "      -b <background-column-name>  (default: BACKGROUND)\n"
		   "      [-d]: sort in descending order (default is ascending)\n"
		   "      [-v]: add verboseness.\n"
           "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	char* progname = args[0];
    char* fluxcol = NULL;
    char* backcol = NULL;
    anbool ascending = TRUE;
	int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'f':
            fluxcol = optarg;
            break;
        case 'b':
            backcol = optarg;
            break;
        case 'd':
            ascending = FALSE;
            break;
		case 'v':
			loglvl++;
			break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }
	log_init(loglvl);

    if (optind != argc-2) {
        printHelp(progname);
        exit(-1);
    }

    infn = args[optind];
    outfn = args[optind+1];

    fits_use_error_system();

    if (resort_xylist(infn, outfn, fluxcol, backcol, ascending)) {
        ERROR("Failed to re-sorting xylist by FLUX and BACKGROUND");
        exit(-1);
    }

    return 0;
}
