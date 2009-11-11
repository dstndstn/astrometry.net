/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2008, 2009 Dustin Lang.

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

#include "sip_qfits.h"
#include "an-bool.h"
#include "qfits.h"
#include "starutil.h"
#include "bl.h"
#include "xylist.h"
#include "rdlist.h"
#include "boilerplate.h"
#include "errors.h"

const char* OPTIONS = "hi:o:w:f:X:Y:tx:y:e:";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   [-e <extension>] FITS HDU number to read WCS from (default 0 = primary)\n"
		   "   -i <xyls input file>\n"
		   "   -o <rdls output file>\n"
		   "  [-f <xyls field index>] (default: all)\n"
		   "  [-X <x-column-name> -Y <y-column-name>]\n"
		   "  [-t]: just use TAN projection, even if SIP extension exists.\n"
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
	double x, y;
	char* xylsfn = NULL;
	char* wcsfn = NULL;
	char* rdlsfn = NULL;
	char* xcol = NULL;
	char* ycol = NULL;
	bool forcetan = FALSE;
	int ext = 0;

	rdlist_t* rdls = NULL;
	xylist_t* xyls = NULL;
	il* fields;
	sip_t sip;
	int i;
	x = y = HUGE_VAL;

	fields = il_new(16);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case 'e':
			ext = atoi(optarg);
			break;
		case 'x':
			x = atof(optarg);
			break;
		case 'y':
			y = atof(optarg);
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

	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}

	if (!(wcsfn && ((rdlsfn && xylsfn) || ((x != HUGE_VAL) && (y != HUGE_VAL))))) {
		print_help(args[0]);
		exit(-1);
	}

	// read WCS.
	if (!sip_read_tan_or_sip_header_file_ext(wcsfn, ext, &sip, forcetan)) {
		ERROR("Failed to read WCS file");
		exit(-1);
	}

	if (!xylsfn) {
		// convert immediately.
		double ra,dec;
		sip_pixelxy2radec(&sip, x, y, &ra, &dec);
		printf("Pixel (%f, %f) -> RA,Dec (%f, %f)\n", x, y, ra, dec);
		exit(0);
	}

	// read XYLS.
	xyls = xylist_open(xylsfn);
	if (!xyls) {
		fprintf(stderr, "Failed to read an xylist from file %s.\n", xylsfn);
		exit(-1);
	}
    xylist_set_include_flux(xyls, FALSE);
    xylist_set_include_background(xyls, FALSE);
	if (xcol)
		xylist_set_xname(xyls, xcol);
	if (ycol)
		xylist_set_yname(xyls, ycol);

	// write RDLS.
	rdls = rdlist_open_for_writing(rdlsfn);
	if (!rdls) {
		fprintf(stderr, "Failed to open file %s to write RDLS.\n", rdlsfn);
		exit(-1);
	}
	if (rdlist_write_primary_header(rdls)) {
		fprintf(stderr, "Failed to write header to RDLS file %s.\n", rdlsfn);
		exit(-1);
	}

	if (!il_size(fields)) {
		// add all fields.
		int NF = xylist_n_fields(xyls);
		for (i=1; i<=NF; i++)
			il_append(fields, i);
	}

	fprintf(stderr, "Processing %i extensions...\n", il_size(fields));
	for (i=0; i<il_size(fields); i++) {
		int fieldind = il_get(fields, i);
        starxy_t xy;
        rd_t rd;
		int j;

        if (!xylist_read_field_num(xyls, fieldind, &xy)) {
			fprintf(stderr, "Failed to read xyls field %i.\n", fieldind);
			exit(-1);
        }

		if (rdlist_write_header(rdls)) {
			fprintf(stderr, "Failed to write rdls field header.\n");
			exit(-1);
		}

        rd_alloc_data(&rd, starxy_n(&xy));

		for (j=0; j<starxy_n(&xy); j++) {
            double x, y, ra, dec;
            x = starxy_getx(&xy, j);
            y = starxy_gety(&xy, j);
			sip_pixelxy2radec(&sip, x, y, &ra, &dec);

            rd_setra (&rd, j, ra);
            rd_setdec(&rd, j, dec);
		}

        if (rdlist_write_field(rdls, &rd)) {
            fprintf(stderr, "Failed to write rdls field.\n");
            exit(-1);
        }
        rd_free_data(&rd);
        starxy_free_data(&xy);

		if (rdlist_fix_header(rdls)) {
			fprintf(stderr, "Failed to fix rdls field header.\n");
			exit(-1);
		}

        rdlist_next_field(rdls);
	}

	if (rdlist_fix_primary_header(rdls) ||
		rdlist_close(rdls)) {
		fprintf(stderr, "Failed to fix header of RDLS file.\n");
		exit(-1);
	}

	if (xylist_close(xyls)) {
		fprintf(stderr, "Failed to close XYLS file.\n");
	}

	fprintf(stderr, "Done!\n");

	return 0;
}
