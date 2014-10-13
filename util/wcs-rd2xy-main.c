/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2010, 2013 Dustin Lang.

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

#include "an-bool.h"
#include "bl.h"
#include "boilerplate.h"
#include "wcs-rd2xy.h"
#include "anwcs.h"
#include "errors.h"
#include "log.h"

const char* OPTIONS = "hi:o:w:f:R:D:te:r:d:Lv";

void print_help(char* progname) {
	BOILERPLATE_HELP_HEADER(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   [-e <extension>] HDU to read (default 0 = primary)\n"
		   "   -i <rdls input file>\n"
		   "   -o <xyls output file>\n"
		   "  [-f <rdls field index>] (default: all)\n"
		   "  [-R <RA-column-name> -D <Dec-column-name>]\n"
		   "  [-t]: just use TAN projection, even if SIP extension exists\n"
		   "  [-L]: force using WCSlib rather than Astrometry.net routines\n"
		   "  [-v]: +verbose\n"
		   "You can also just specify a single point to convert (printed to stdout)\n"
		   "   [-r <ra>], RA in deg.\n"
		   "   [-d <ra>], Dec in deg.\n"
		   "\n", progname);
}


int main(int argc, char** args) {
	int c;
	char* rdlsfn = NULL;
	char* wcsfn = NULL;
	char* xylsfn = NULL;
	char* rcol = NULL;
	char* dcol = NULL;
	anbool forcetan = FALSE;
	il* fields;
	int ext = 0;
	double ra=HUGE_VAL, dec=HUGE_VAL;
	anbool wcslib = FALSE;
	int loglvl = LOG_MSG;

	fields = il_new(16);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case 'v':
			loglvl++;
			break;
		case 'L':
			wcslib = TRUE;
			break;
		case 'r':
			ra = atof(optarg);
			break;
		case 'd':
			dec = atof(optarg);
			break;
		case 'e':
			ext = atoi(optarg);
			break;
        case 'h':
			print_help(args[0]);
			exit(0);
		case 't':
			forcetan = TRUE;
			break;
		case 'o':
			xylsfn = optarg;
			break;
		case 'i':
			rdlsfn = optarg;
			break;
		case 'w':
			wcsfn = optarg;
			break;
		case 'f':
			il_append(fields, atoi(optarg));
			break;
		case 'R':
			rcol = optarg;
			break;
		case 'D':
			dcol = optarg;
			break;
		}
	}

	log_init(loglvl);

	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}

	if (!(wcsfn && ((rdlsfn && xylsfn) || ((ra != HUGE_VAL) && (dec != HUGE_VAL))))) {
		print_help(args[0]);
		exit(-1);
	}

	if (!rdlsfn) {
		double x,y;
		anwcs_t* wcs = NULL;

		// read WCS.
		if (wcslib) {
			wcs = anwcs_open_wcslib(wcsfn, ext);
		} else if (forcetan) {
			wcs = anwcs_open_tan(wcsfn, ext);
		} else {
			wcs = anwcs_open(wcsfn, ext);
		}
		if (!wcs) {
			ERROR("Failed to read WCS file");
			exit(-1);
		}
		logverb("Read WCS:\n");
		if (log_get_level() >= LOG_VERB) {
			anwcs_print(wcs, log_get_fid());
		}

		// convert immediately.
		if (anwcs_radec2pixelxy(wcs, ra, dec, &x, &y)) {
			ERROR("The given RA,Dec is on the opposite side of the sky.");
			exit(-1);
		}
		printf("RA,Dec (%.10f, %.10f) -> pixel (%.10f, %.10f)\n", ra, dec, x, y);
		anwcs_free(wcs);
		exit(0);
	}


    if (wcs_rd2xy(wcsfn, ext, rdlsfn, xylsfn,
                  rcol, dcol, forcetan, wcslib, fields)) {
        ERROR("wcs-rd2xy failed");
        exit(-1);
    }

	return 0;
}
