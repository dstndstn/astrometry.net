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
#include <math.h>
#include <string.h>

#include "kdtree.h"
#include "starutil.h"
#include "quadfile.h"
#include "fitsioutils.h"
#include "qfits.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "unpermute-stars.h"

static const char* OPTIONS = "hs:q:S:Q:wcv";

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
           "    -s <input-star-kdtree-filename>\n"
           "    -q <input-quads-filename>\n"
           "    -S <output-star-kdtree-filename>\n"
           "    -Q <output-quads-filename>\n"
		   "   [-w]: store sweep number in output star kdtree file.\n"
		   "   [-c]: check values\n"
		   "   [-v]: more verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **args) {
    int argchar;
	char* progname = args[0];
    char* quadinfn = NULL;
    char* skdtinfn = NULL;
    char* quadoutfn = NULL;
    char* skdtoutfn = NULL;
	anbool dosweeps = FALSE;
	anbool check = FALSE;
	int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'c':
			check = TRUE;
			break;
		case 'v':
			loglvl++;
			break;
		case 'q':
			quadinfn = optarg;
			break;
		case 'Q':
			quadoutfn = optarg;
			break;
        case 's':
            skdtinfn = optarg;
            break;
        case 'S':
            skdtoutfn = optarg;
            break;
		case 'w':
			dosweeps = TRUE;
			break;
        case '?':
            ERROR("Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	log_init(loglvl);

	if (!(quadinfn && quadoutfn && skdtinfn && skdtoutfn)) {
		printHelp(progname);
        ERROR("\nMust include all filenames (-q, -Q, -s, -S)\n");
		exit(-1);
	}

	if (unpermute_stars_files(skdtinfn, quadinfn, skdtoutfn, quadoutfn,
							  dosweeps, check, args, argc)) {
		exit(-1);
	}
	return 0;
}
