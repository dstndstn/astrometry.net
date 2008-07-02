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

#include "starutil.h"
#include "catalog.h"

#define OPTIONS "hn:f:r:R:d:D:S:"
const char HelpString[] =
  "randcat -f fname -n numstars [-r/R RAmin/max] [-d/D DECmin/max]\n"
    "  -r -R -d -D set ra and dec limits in radians\n";

extern char *optarg;
extern int optind, opterr, optopt;

int RANDSEED = 999;

int main(int argc, char *argv[])
{
	int argidx, argchar;
	int numstars = 10;
	char *fname = NULL;
	double ramin = DEFAULT_RAMIN, ramax = DEFAULT_RAMAX;
	double decmin = DEFAULT_DECMIN, decmax = DEFAULT_DECMAX;
	int i;
	catalog* cat;

	if (argc <= 4) {
		fprintf(stderr, HelpString);
		return (OPT_ERR);
	}

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'S':
			RANDSEED = atoi(optarg);
			break;
		case 'n':
			numstars = strtoul(optarg, NULL, 0);
			break;
		case 'r':
			ramin = strtod(optarg, NULL);
			break;
		case 'R':
			ramax = strtod(optarg, NULL);
			break;
		case 'd':
			decmin = strtod(optarg, NULL);
			break;
		case 'D':
			decmax = strtod(optarg, NULL);
			break;
		case 'f':
			fname = mk_catfn(optarg);
			break;
		case '?':
			fprintf (stderr, "Unknown option `-%c'.\n", optopt);
		case 'h':
			fprintf(stderr, HelpString);
			return (HELP_ERR);
		default:
			return (OPT_ERR);
		}

	if (optind < argc) {
		for (argidx = optind; argidx < argc; argidx++)
			fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
		fprintf(stderr, HelpString);
		return (OPT_ERR);
	}

	srand(RANDSEED);

	fprintf(stderr, "randcat: Making %u random stars\n", numstars);
	fprintf(stderr, "[RANDSEED=%d]\n", RANDSEED);
	if (ramin > DEFAULT_RAMIN || ramax < DEFAULT_RAMAX ||
		decmin > DEFAULT_DECMIN || decmax < DEFAULT_DECMAX)
		fprintf(stderr, "  using limits %f<=RA<=%f ; %f<=DEC<=%f deg.\n",
				rad2deg(ramin), rad2deg(ramax), rad2deg(decmin), rad2deg(decmax));

    cat = catalog_open_for_writing(fname);

	for (i=0; i <numstars; i++) {
        double star[DIM_STARS];
		make_rand_star(star, ramin, ramax, decmin, decmax);
        if (catalog_write_star(cat, star)) {
            fprintf(stderr, "Failed to write star %i.\n", i);
            exit(-1);
        }
	}

	//catalog_compute_radecminmax(&cat);

    if (catalog_close(cat)) {
		fprintf(stderr, "Failed to write catalog.\n");
		exit(-1);
	}

    return 0;
}



