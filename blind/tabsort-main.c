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

#include "tabsort.h"
#include "fitsioutils.h"

static const char* OPTIONS = "hd";

static void printHelp(char* progname) {
	printf("%s  [options]  <column-name> <input-file> <output-file>\n"
           "  options include:\n"
		   "      [-d]: sort in descending order (default, ascending)\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	char* colname = NULL;
	char* progname = argv[0];
	anbool descending = FALSE;

    while ((argchar = getopt(argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'd':
			descending = TRUE;
			break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (optind != argc-3) {
		printHelp(progname);
		exit(-1);
    }

    colname = argv[optind  ];
    infn    = argv[optind+1];
    outfn   = argv[optind+2];

    fits_use_error_system();

    return tabsort(infn, outfn, colname, descending);
}

