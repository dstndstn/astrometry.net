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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include "qfits.h"
#include "sip.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "starutil.h"

static char* OPTIONS = "h";

static void printHelp(char* progname) {
	printf("%s  <input-file>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
	char* progname = argv[0];
    int argchar;
	char* infn = NULL;
	qfits_header* hdr;
	sip_t sip;
	double val;
	bool gotsip = FALSE;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (optind != (argc - 1)) {
		printHelp(progname);
		exit(-1);
	}
	infn = argv[optind];

	hdr = qfits_header_read(infn);
	if (!hdr) {
		fprintf(stderr, "Failed to read FITS header.\n");
		exit(-1);
	}

	memset(&sip, 0, sizeof(sip_t));
	if (sip_read_header(hdr, &sip)) {
		double det = sip_det_cd(&sip);
		if (det != 0) {
			printf("scale sip %g\n", 3600.0 * sqrt(fabs(det)));
			// CHECK: printf("sip_parity %g\n", det > 0 ? 1 : 0);
			gotsip = TRUE;
		}
	}

	if (!gotsip) {
		// it might have a correct CD matrix but be missing other parts (eg CRVAL)
		double cd11, cd12, cd21, cd22;
		double errval = -HUGE_VAL;
		cd11 = qfits_header_getdouble(hdr, "CD1_1", errval);
		cd12 = qfits_header_getdouble(hdr, "CD1_2", errval);
		cd21 = qfits_header_getdouble(hdr, "CD2_1", errval);
		cd22 = qfits_header_getdouble(hdr, "CD2_2", errval);
		if ((cd11 != errval) && (cd12 != errval) && (cd21 != errval) && (cd22 != errval)) {
			double det = cd11 * cd22 - cd12 * cd21;
			if (det != 0) {
				printf("scale cd %g\n", 3600.0 * sqrt(fabs(det)));
			}
		}
	}

	val = qfits_header_getdouble(hdr, "PIXSCALE", -1.0);
	if (val != -1.0) {
		printf("scale pixscale %g\n", val);
	}

	val = qfits_header_getdouble(hdr, "PIXSCAL1", -1.0);
	if (val != -1.0) {
		if (val != 0.0) {
			printf("scale pixscal1 %g\n", val);
		} else {
			val = atof(qfits_pretty_string(qfits_header_getstr(hdr, "PIXSCAL1")));
			if (val != 0.0) {
				printf("scale pixscal1 %g\n", val);
			}
		}
	}

	val = qfits_header_getdouble(hdr, "PIXSCAL2", -1.0);
	if (val != -1.0) {
		if (val != 0.0) {
			printf("scale pixscal2 %g\n", val);
		} else {
			val = atof(qfits_pretty_string(qfits_header_getstr(hdr, "PIXSCAL2")));
			if (val != 0.0) {
				printf("scale pixscal2 %g\n", val);
			}
		}
	}

	val = qfits_header_getdouble(hdr, "PLATESC", -1.0);
	if ((val != -1.0) && (val != 0.0)) {
		printf("scale platesc %g\n", val);
	}

	val = qfits_header_getdouble(hdr, "CCDSCALE", -1.0);
	if ((val != -1.0) && (val != 0.0)) {
		printf("scale ccdscale %g\n", val);
	}

	val = qfits_header_getdouble(hdr, "CDELT1", -1.0);
	if ((val != -1.0) && (val != 0.0)) {
		printf("scale cdelt1 %g\n", 3600.0 * fabs(val));
	}

	return 0;
}
