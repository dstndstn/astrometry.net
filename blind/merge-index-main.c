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
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "fitsioutils.h"
#include "errors.h"
#include "boilerplate.h"
#include "ioutils.h"
#include "merge-index.h"

#define OPTIONS "hq:c:s:o:"

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
           "   -q <input-quad-filename>\n"
           "   -c <input-code-kdtree-filename>\n"
           "   -s <input-star-kdtree-filename>\n"
           "   -o <output-index-filename>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **args) {
    int argchar;
	char* progname = args[0];
	char* quadfn = NULL;
	char* codefn = NULL;
	char* starfn = NULL;
	char* outfn = NULL;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'q':
            quadfn = optarg;
            break;
        case 'c':
            codefn = optarg;
            break;
        case 's':
            starfn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!(quadfn && starfn && codefn && outfn)) {
		printHelp(progname);
        fprintf(stderr, "\nYou must specify all filenames (-q, -c, -s, -o)\n");
		exit(-1);
	}

    fits_use_error_system();

	if (merge_index_files(quadfn, codefn, starfn, outfn)) {
		exit(-1);
	}
	return 0;
}
