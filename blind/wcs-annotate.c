/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "sip_qfits.h"
#include "an-bool.h"
#include "qfits.h"
#include "starutil.h"
#include "mathutil.h"
#include "boilerplate.h"

#include "ngc2000.h"


const char* OPTIONS = "hw:p";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   [-p]: give pixel locations\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* wcsfn = NULL;
	sip_t sip;
	bool hassip = FALSE;
	int i;
	int N;
    int W, H;
	double scale;
	double imsize;
	bool pix = FALSE;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'w':
			wcsfn = optarg;
			break;
		case 'p':
			pix = TRUE;
			break;
		}
	}

	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}

	if (!wcsfn) {
		print_help(args[0]);
		exit(-1);
	}

	// read WCS.
	fprintf(stderr, "Trying to parse SIP header from %s...\n", wcsfn);
	if (sip_read_header_file(wcsfn, &sip)) {
		fprintf(stderr, "Got SIP header.\n");
		hassip = (sip.a_order > 0);
	} else {
		fprintf(stderr, "Failed to parse SIP or TAN header from %s.\n", wcsfn);
		exit(-1);
	}

    if ((sip.wcstan.imagew == 0.0) || (sip.wcstan.imageh == 0.0)) {
		fprintf(stderr, "IMAGEW, IMAGEH FITS headers not found.\n");
		exit(-1);
	}
    W = sip.wcstan.imagew;
    H = sip.wcstan.imageh;

	// arcsec/pixel
	scale = sip_pixel_scale(&sip);

	// arcmin
	imsize = scale * MIN(W, H) / 60.0;

	N = ngc_num_entries();
	for (i=0; i<N; i++) {
		ngc_entry* ngc;
		sl* names;
		double ra, dec;
		double x,y;
		int n;

		ngc = ngc_get_entry(i);
		if (!ngc)
			break;

		// If the image is way bigger than the NGC object...
		// (I think NGC objects that don't actually exist have size=0.)
		if (ngc->size < imsize * 0.02)
			continue;

		ra = ngc->ra;
		dec = ngc->dec;

		if (!sip_radec2pixelxy(&sip, ra, dec, &x, &y))
			continue;

		if (x < 0 || y < 0 || x > W || y > H)
			continue;

		printf("%s %i", (ngc->is_ngc ? "NGC" : "IC"), ngc->id);

		names = ngc_get_names(ngc);
		if (names) {
			for (n=0; n<sl_size(names); n++)
				printf(" / %s", sl_get(names, n));
			sl_free2(names);
		}
		if (pix)
			printf(" near pixel (%i, %i)\n", (int)round(x), (int)round(y));
		printf("\n");
	}

	fprintf(stderr, "Done!\n");

	return 0;
}
