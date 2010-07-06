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
		   "    -r <ra>     (deg)\n"
		   "    -d <dec>    (deg)\n"
		   "    -R <radius> (deg)\n"
		   "    [-v]: +verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **argv) {
    int argchar;
	double ra=HUGE_VAL, dec=HUGE_VAL, radius=HUGE_VAL;
	int loglvl = LOG_MSG;
	char** myargs;
	int nmyargs;
	int i;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'r':
			ra = atof(optarg);
			break;
		case 'd':
			dec = atof(optarg);
			break;
		case 'R':
			radius = atof(optarg);
			break;
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
	if (ra == HUGE_VAL || dec == HUGE_VAL || radius == HUGE_VAL) {
		printHelp(argv[0]);
		exit(-1);
	}

	for (i=0; i<nmyargs; i++) {
		char* indexfn = myargs[i];
		index_t index;
		sl* cols;
		int* inds;
		double* radecs;
		int N;
		int j;

		/*
		 tic();
		 logmsg("Reading meta-data for index %s\n", indexfn);
		 if (index_get_meta(indexfn, &index)) {
		 ERROR("Failed to read metadata for index %s", indexfn);
		 continue;
		 }
		 toc();
		 */

		if (!index_load(indexfn, 0, &index)) {
			ERROR("Failed to read index \"%s\"", indexfn);
			continue;
		}

		logmsg("Index %s: id %i, healpix %i (nside %i), %i stars, %i quads, dimquads=%i, scales %g to %g arcmin.\n",
			   index.indexname,
			   index.indexid, index.healpix, index.hpnside,
			   index.nstars, index.nquads, index.dimquads,
			   arcsec2arcmin(index.index_scale_lower),
			   arcsec2arcmin(index.index_scale_upper));

		cols = startree_get_tagalong_column_names(index.starkd, NULL);
		{
			char* colstr = sl_join(cols, ", ");
			logmsg("Tag-along columns: %s\n", colstr);
			free(colstr);
		}

		logmsg("Searching for stars around RA,Dec (%g, %g), radius %g deg.\n",
			   ra, dec, radius);
		startree_search_for_radec(index.starkd, ra, dec, radius,
								  NULL, &radecs, &inds, &N);
		logmsg("Found %i stars\n", N);

		for (j=0; j<sl_size(cols); j++) {
			char* col;
			double* dat;
			col = sl_get(cols, j);
			logmsg("Grabbing tag-along column \"%s\"...\n", col);
			dat = startree_get_data_column(index.starkd, col, inds, N);
			// ...
			free(dat);
		}

		sl_free2(cols);
		free(radecs);
		free(inds);

		index_close(&index);
	}

	return 0;
}

