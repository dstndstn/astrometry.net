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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/param.h>

#include "index.h"
#include "quadfile.h"
#include "kdtree.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "histogram.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"

static const char* OPTIONS = "hvmM:";


void print_help(char* progname) {
	BOILERPLATE_HELP_HEADER(stderr);
	fprintf(stderr, "Usage: %s\n"
			"   [-h]: help\n"
			"   [-m]: measure range of mags of quad stars.\n"
			"   [-M <column-name>]: set name of magnitude column.\n"
			"   <index-base-name> [<index-base-name> ...]\n\n"
			"\n", progname);
}

int main(int argc, char** args) {
    int argchar;
	anbool measmags = FALSE;
	char* magcol = "mag";
	int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'm':
			measmags = TRUE;
			break;
		case 'M':
			magcol = optarg;
			break;
		case 'h':
			print_help(args[0]);
			exit(0);
		case 'v':
			loglvl++;
			break;
		}
	log_init(loglvl);
	log_to(stderr);

	if (optind == argc) {
		print_help(args[0]);
		exit(-1);
	}

	for (; optind<argc; optind++) {
		int lastgrass = 0;
        int dimquads;
		char* basename;
		index_t* index;
		int i, N, k;

		basename = args[optind];
		logmsg("Reading index %s\n", basename);

		index = index_load(basename, 0, NULL);

		dimquads = index_get_quad_dim(index);
		N = index_nquads(index);
		logmsg("%i quads.\n", N);

		if (measmags) {
			printf("# mag_mean mag_var mag_min mag_max");
			for (k=0; k<dimquads; k++)
				printf(" mag_%i", k);
			printf("\n");
			for (i=0; i<N; i++) {
				int grass;
				unsigned int stars[DQMAX];
				double* mags;
				double mean, var, mn, mx;
				grass = (i * 80 / N);
				if (grass != lastgrass) {
					fprintf(stderr, ".");
					fflush(stderr);
					lastgrass = grass;
				}
				quadfile_get_stars(index->quads, i, stars);

				mags = startree_get_data_column(index->starkd, magcol, stars, dimquads);
				mean = var = 0.0;
				for (k=0; k<dimquads; k++) 
					mean += mags[k];
				mean /= (double)dimquads;
				for (k=0; k<dimquads; k++) 
					var += square(mags[k] - mean);
				var /= (double)dimquads;
				mn =  HUGE_VAL;
				mx = -HUGE_VAL;
				for (k=0; k<dimquads; k++) {
					mn = MIN(mn, mags[k]);
					mx = MAX(mx, mags[k]);
				}
					
				printf("%g %g %g %g", mean, var, mn, mx);
				for (k=0; k<dimquads; k++)
					printf(" %g", mags[k]);
				printf("\n");

				startree_free_data_column(index->starkd, mags);
				/*
				 for (k=0; k<dimquads; k++)
				 startree_get_radec(index->starkd, stars[k], &ra, &dec);
				 */
			}
		}
		fprintf(stderr, "\n");

		index_close(index);
	}
	return 0;
}
