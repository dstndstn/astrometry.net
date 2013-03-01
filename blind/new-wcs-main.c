/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include <sys/types.h>

#include "new-wcs.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"

static const char* OPTIONS = "hi:w:o:d";

static void printHelp(char* progname) {
	printf("%s    -i <input-file>\n"
		   "      -w <WCS-file>\n"
		   "      -o <output-file>\n"
           "      [-d]: also copy the data segment\n"
		   "\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	char* wcsfn = NULL;
	char* progname = argv[0];
    anbool copydata = FALSE;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'i':
			infn = optarg;
			break;
        case 'o':
			outfn = optarg;
			break;
        case 'w':
			wcsfn = optarg;
			break;
        case 'd':
            copydata = TRUE;
            break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!infn || !outfn || !wcsfn) {
		printHelp(progname);
		exit(-1);
	}

    fits_use_error_system();

    if (new_wcs(infn, wcsfn, outfn, copydata)) {
        ERROR("new_wcs() failed");
        exit(-1);
    }
    return 0;
}
