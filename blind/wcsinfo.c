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
#include <errno.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "sip.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "mathutil.h"
#include "boilerplate.h"

const char* OPTIONS = "h";

void printHelp(char* progname) {
	boilerplate_help_header(stderr);
	fprintf(stderr, "\nUsage: %s <wcs-file>\n"
			"\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int argchar;
	char* progname = args[0];
	char** inputfiles = NULL;
	int ninputfiles = 0;
	sip_t wcs;
	double imw, imh;
	double rac, decc;
	double det, T, A, parity, orient;
    int rah, ram, decd, decm;
    double ras, decs;
    char* units;
    double pixscale;
    double fldw, fldh;
    double ramin, ramax, decmin, decmax;
	//double mxlo, mxhi, mylo, myhi;
	double dm;
	int merczoom;
    char rastr[32];
    char decstr[32];

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

	if (!sip_read_header_file(inputfiles[0], &wcs)) {
		fprintf(stderr, "failed to read WCS header from file %s.\n", inputfiles[0]);
		return -1;
	}

    imw = wcs.wcstan.imagew;
    imh = wcs.wcstan.imageh;
	if ((imw == 0.0) || (imh == 0.0)) {
		fprintf(stderr, "failed to find IMAGE{W,H} in WCS file.\n");
		return -1;
	}

    printf("imagew %.12g\n", imw);
    printf("imageh %.12g\n", imh);

	printf("cd11 %.12g\n", wcs.wcstan.cd[0][0]);
	printf("cd12 %.12g\n", wcs.wcstan.cd[0][1]);
	printf("cd21 %.12g\n", wcs.wcstan.cd[1][0]);
	printf("cd22 %.12g\n", wcs.wcstan.cd[1][1]);

	det = sip_det_cd(&wcs);
	parity = (det >= 0 ? 1.0 : -1.0);
    pixscale = sip_pixel_scale(&wcs);
	printf("det %.12g\n", det);
	printf("parity %i\n", (int)parity);
	printf("pixscale %.12g\n", pixscale);

	T = parity * wcs.wcstan.cd[0][0] + wcs.wcstan.cd[1][1];
	A = parity * wcs.wcstan.cd[1][0] - wcs.wcstan.cd[0][1];
	orient = -rad2deg(atan2(A, T));
	printf("orientation %.8g\n", orient);

    sip_get_radec_center(&wcs, &rac, &decc);
	printf("ra_center %.12g\n", rac);
	printf("dec_center %.12g\n", decc);

    sip_get_radec_center_hms(&wcs, &rah, &ram, &ras, &decd, &decm, &decs);
    printf("ra_center_h %i\n", rah);
    printf("ra_center_m %i\n", ram);
    printf("ra_center_s %.12g\n", ras);
    printf("dec_center_d %i\n", decd);
    printf("dec_center_m %i\n", decm);
    printf("dec_center_s %.12g\n", decs);

    sip_get_radec_center_hms_string(&wcs, rastr, decstr);
    printf("ra_center_hms %s\n", rastr);
    printf("dec_center_dms %s\n", decstr);

	// mercator
	printf("ra_center_merc %.8g\n", ra2mercx(rac));
	printf("dec_center_merc %.8g\n", dec2mercy(decc));

    fldw = imw * pixscale;
    fldh = imh * pixscale;
    // area of the field, in square degrees.
    printf("fieldarea %g\n", (arcsec2deg(fldw) * arcsec2deg(fldh)));

    sip_get_field_size(&wcs, &fldw, &fldh, &units);
    printf("fieldw %.4g\n", fldw);
    printf("fieldh %.4g\n", fldh);
    printf("fieldunits %s\n", units);

    sip_get_radec_bounds(&wcs, 10, &ramin, &ramax, &decmin, &decmax);
    printf("decmin %g\n", decmin);
    printf("decmax %g\n", decmax);
    printf("ramin %g\n", ramin);
    printf("ramax %g\n", ramax);

	// merc zoom level
	/*
	  mxlo = ra2mercx(ramax);
	  mxhi = ra2mercx(ramin);
	  mylo = dec2mercy(decmax);
	  myhi = dec2mercy(decmin);
	*/
	dm = MAX(fabs(ra2mercx(ramax) - ra2mercx(ramin)), fabs(dec2mercy(decmax) - dec2mercy(decmin)));
	merczoom = 0 - (int)floor(log(dm) / log(2.0));
	//printf("dm %g\n", dm);
	printf("merczoom %i\n", merczoom);
	return 0;
}
