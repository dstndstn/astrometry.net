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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "starutil.h"
#include "mathutil.h"
#include "boilerplate.h"
#include "rdlist.h"

const char* OPTIONS = "h";

void printHelp(char* progname) {
	boilerplate_help_header(stderr);
	fprintf(stderr, "\nUsage: %s <rdls-file>\n"
			"\n", progname);
}

static double mercy2dec(double y) {
	return rad2deg(atan(sinh((y - 0.5) * (2.0 * M_PI))));
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int argchar;
	char* progname = args[0];
	char** inputfiles = NULL;
	int ninputfiles = 0;
	rdlist_t* rdls;
	rd_t* rd;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1) {
		switch (argchar) {
		case 'h':
		default:
			printHelp(progname);
			exit(-1);
		}
	}
	if (optind < argc) {
		ninputfiles = argc - optind;
		inputfiles = args + optind;
	}
	if (ninputfiles != 1) {
		printHelp(progname);
		exit(-1);
	}

	rdls = rdlist_open(inputfiles[0]);
	if (!rdls) {
		fprintf(stderr, "Failed to open rdls file %s.\n", inputfiles[0]);
		exit(-1);
	}

	rd = rdlist_read_field(rdls, NULL);
	if (!rd) {
		fprintf(stderr, "Failed to get RDLS field.\n");
		exit(-1);
	}

	{
		double ramax, ramin, decmax, decmin;
		double racenter, deccenter;
		double ymerccenter;
		double dx, dy;
		double maxd;
		double xyz1[3], xyz2[3];
		double arc;
		int i;
		ramax = decmax = -HUGE_VAL;
		ramin = decmin =  HUGE_VAL;
		for (i=0; i<rd_n(rd); i++) {
			double ra  = rd_getra (rd, i);
			double dec = rd_getdec(rd, i);
			if (ra > ramax) ramax = ra;
			if (ra < ramin) ramin = ra;
			if (dec > decmax) decmax = dec;
			if (dec < decmin) decmin = dec;
		}
		if (ramax - ramin > 180.0) {
			// probably wrapped around
			ramax = 0.0;
			ramin = 360.0;
			for (i=0; i<rd_n(rd); i++) {
				double ra  = rd_getra(rd, i);
				if (ra > 180)
					ramin = MIN(ramin, ra);
				else
					ramax = MAX(ramax, ra);
			}
			racenter = (ramax - 360 + ramin) / 2.0;
			if (racenter < 0.0)
				racenter += 360.0;
		} else {
			racenter  = (ramin  + ramax ) / 2.0;
		}
		deccenter = (decmin + decmax) / 2.0;
		ymerccenter = mercy2dec((dec2mercy(decmin) + dec2mercy(decmax))/2.0);

		radecdeg2xyzarr(ramin, decmin, xyz1);
		radecdeg2xyzarr(ramax, decmax, xyz2);
		arc = rad2deg(distsq2arc(distsq(xyz1, xyz2, 3) / 2.0));

		printf("ra_min %g\n", ramin);
		printf("ra_max %g\n", ramax);
		printf("dec_min %g\n", decmin);
		printf("dec_max %g\n", decmax);
		printf("ra_center %g\n", racenter);
		printf("dec_center %g\n", deccenter);

		if (arc >= 1) {
			printf("field_size %.3g degrees\n", arc);
		} else {
			arc *= 60.0;
			if (arc >= 1) {
				printf("field_size %.3g arcminutes\n", arc);
			} else {
				arc *= 60.0;
				printf("field_size %.3g arcseconds\n", arc);
			}
		}

		// mercator
		printf("ra_center_merc %g\n", racenter);
		printf("dec_center_merc %g\n", ymerccenter);
		// mercator/gmaps zoomlevel.
		dx = (ra2mercx(ramax) - ra2mercx(ramin));
		dy = (dec2mercy(decmax) - dec2mercy(decmin));
		maxd = (dx > dy ? dx : dy);
		//printf("zoom_merc %i\n", 1 + (int)floor(log(1.0 / maxd) / log(2.0)));
		printf("zoom_merc %i\n", (int)floor(log((600.0/256.0) * 1.0 / maxd) / log(2.0)));
	}

	rd_free(rd);
	rdlist_close(rdls);
	return 0;
}
