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
#include "wcs-xy2rd.h"
#include "log.h"

const char* OPTIONS = "hi:o:w:f:R:D:te:x:y:X:Y:LTv";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   [-e <extension>] FITS HDU number to read WCS from (default 0 = primary)\n"
		   "   [-t]: just use TAN projection, even if SIP extension exists.\n"
		   "   [-L]: force WCSlib\n"
		   "   [-T]: force WCStools\n"
		   "   -i <xyls input file>\n"
		   "   -o <rdls output file>\n"
		   "  [-f <xyls field index>] (default: all)\n"
		   "  [-X <x-column-name> -Y <y-column-name>]\n"
		   "  [-v]: +verbose\n"
		   "\n"
		   "You can also specify a single point to convert (result is printed to stdout):\n"
		   "  [-x <pixel>]\n"
		   "  [-y <pixel>]\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* rdlsfn = NULL;
	char* wcsfn = NULL;
	char* xylsfn = NULL;
	char* xcol = NULL;
	char* ycol = NULL;
	anbool forcetan = FALSE;
	anbool forcewcslib = FALSE;
	anbool forcewcstools = FALSE;
	il* fields;
	int ext = 0;
	double x, y;
	int loglvl = LOG_MSG;

	x = y = HUGE_VAL;
	fields = il_new(16);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case 'v':
			loglvl++;
			break;
		case 'L':
			forcewcslib = TRUE;
			break;
		case 'T':
			forcewcstools = TRUE;
			break;
		case 'x':
			x = atof(optarg);
			break;
		case 'y':
			y = atof(optarg);
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
			rdlsfn = optarg;
			break;
		case 'i':
			xylsfn = optarg;
			break;
		case 'w':
			wcsfn = optarg;
			break;
		case 'f':
			il_append(fields, atoi(optarg));
			break;
		case 'X':
			xcol = optarg;
			break;
		case 'Y':
			ycol = optarg;
			break;
		}
	}

	log_init(loglvl);

	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}

	if (!(wcsfn && ((rdlsfn && xylsfn) || ((x != HUGE_VAL) && (y != HUGE_VAL))))) {
		print_help(args[0]);
		exit(-1);
	}

	if (!xylsfn) {
		double ra,dec;
		anwcs_t* wcs = NULL;

		// read WCS.
		if (forcewcslib) {
			wcs = anwcs_open_wcslib(wcsfn, ext);
		} else if (forcewcstools) {
			wcs = anwcs_open_wcstools(wcsfn, ext);
		} else if (forcetan) {
			wcs = anwcs_open_tan(wcsfn, ext);
		} else {
			wcs = anwcs_open(wcsfn, ext);
		}
		if (!wcs) {
			ERROR("Failed to read WCS file \"%s\", extension %i", wcsfn, ext);
			exit(-1);
		}

		logverb("Read WCS:\n");
		if (log_get_level() >= LOG_VERB) {
			anwcs_print(wcs, log_get_fid());
		}

		// convert immediately.
		anwcs_pixelxy2radec(wcs, x, y, &ra, &dec);
		printf("Pixel (%.10f, %.10f) -> RA,Dec (%.10f, %.10f)\n", x, y, ra, dec);
		anwcs_free(wcs);
		exit(0);
	}

    if (wcs_xy2rd(wcsfn, ext, xylsfn, rdlsfn, 
                  xcol, ycol, forcetan, forcewcslib, fields)) {
        ERROR("wcs-xy2rd failed");
        exit(-1);
    }

	return 0;
}
