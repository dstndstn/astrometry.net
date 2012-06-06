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

#include <stdlib.h>
#include <math.h>

#include "index.h"
#include "starutil.h"
#include "log.h"
#include "errors.h"
#include "ioutils.h"
#include "boilerplate.h"
#include "tic.h"

static const char* OPTIONS = "hvr:d:R:";

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] <index-files>\n"
		   "    [-r <ra>] (deg)\n"
		   "    [-d <dec>] (deg)\n"
		   "    [-R <radius>] (deg)\n"
		   "    [-v]: +verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **argv) {
    int argchar;
	//double ra=HUGE_VAL, dec=HUGE_VAL, radius=HUGE_VAL;
	int loglvl = LOG_MSG;
	char** myargs;
	int nmyargs;
	int i;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
			/*
		case 'r':
			ra = atof(optarg);
			break;
		case 'd':
			dec = atof(optarg);
			break;
		case 'R':
			radius = atof(optarg);
			break;
			 */
		case 'v':
			loglvl++;
			break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
		case 'h':
			printHelp(argv[0]);
			break;
		default:
			return -1;
		}
	log_init(loglvl);
	nmyargs = argc - optind;
	myargs = argv + optind;

	if (nmyargs < 1) {
		printHelp(argv[0]);
		exit(-1);
	}

	for (i=0; i<nmyargs; i++) {
		char* indexfn = myargs[i];
		index_t index;
		tic();
		logmsg("Reading meta-data for index %s\n", indexfn);
		if (index_get_meta(indexfn, &index)) {
			ERROR("Failed to read metadata for index %s", indexfn);
			continue;
		}
		toc();

		logmsg("Index %s: id %i, healpix %i (nside %i), %i stars, %i quads, dimquads=%i, scales %g to %g arcmin.\n",
			   index.indexname,
			   index.indexid, index.healpix, index.hpnside,
			   index.nstars, index.nquads, index.dimquads,
			   arcsec2arcmin(index.index_scale_lower),
			   arcsec2arcmin(index.index_scale_upper));

		index_close(&index);
	}

	return 0;
}

