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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "sip_qfits.h"
#include "an-bool.h"
#include "qfits.h"
#include "starutil.h"
#include "bl.h"
#include "xylist.h"
#include "rdlist.h"
#include "boilerplate.h"

const char* OPTIONS = "hi:o:w:f:R:D:tq";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   -i <rdls input file>\n"
		   "   -o <xyls output file>\n"
		   "  [-f <rdls field index>] (default: all)\n"
		   "  [-R <RA-column-name> -D <Dec-column-name>]\n"
		   "  [-t]: just use TAN projection, even if SIP extension exists.\n"
           "  [-q]: quiet\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* rdlsfn = NULL;
	char* wcsfn = NULL;
	char* xylsfn = NULL;
	char* rcol = NULL;
	char* dcol = NULL;
	bool forcetan = FALSE;
    bool verbose = TRUE;

	xylist_t* xyls = NULL;
	rdlist_t* rdls = NULL;
	il* fields;
	sip_t sip;
	int i;

	fields = il_new(16);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
        case 'q':
            verbose = FALSE;
            break;
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

	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}

	if (!xylsfn || !rdlsfn || !wcsfn) {
		print_help(args[0]);
		exit(-1);
	}

	// read WCS.
	if (forcetan) {
		memset(&sip, 0, sizeof(sip_t));
		if (!tan_read_header_file(wcsfn, &(sip.wcstan))) {
			fprintf(stderr, "Failed to parse TAN header from %s.\n", wcsfn);
			exit(-1);
		}
	} else {
		if (!sip_read_header_file(wcsfn, &sip)) {
			printf("Failed to parse SIP header from %s.\n", wcsfn);
			exit(-1);
		}
	}

	// read RDLS.
	rdls = rdlist_open(rdlsfn);
	if (!rdls) {
		fprintf(stderr, "Failed to read an rdlist from file %s.\n", rdlsfn);
		exit(-1);
	}
	if (rcol)
        rdlist_set_raname(rdls, rcol);
	if (dcol)
		rdlist_set_decname(rdls, dcol);

	// write XYLS.
	xyls = xylist_open_for_writing(xylsfn);
	if (!xyls) {
		fprintf(stderr, "Failed to open file %s to write XYLS.\n", xylsfn);
		exit(-1);
	}
	if (xylist_write_primary_header(xyls)) {
		fprintf(stderr, "Failed to write header to XYLS file %s.\n", xylsfn);
		exit(-1);
	}

	if (!il_size(fields)) {
		// add all fields.
		int NF = rdlist_n_fields(rdls);
		for (i=1; i<=NF; i++)
			il_append(fields, i);
	}

	if (verbose)
        printf("Processing %i extensions...\n", il_size(fields));
	for (i=0; i<il_size(fields); i++) {
		int fieldnum = il_get(fields, i);
		int j;
        xy_t xy;
        rd_t rd;

        if (!rdlist_read_field_num(rdls, fieldnum, &rd)) {
			fprintf(stderr, "Failed to read rdls field %i.\n", fieldnum);
			exit(-1);
        }

        xy_alloc_data(&xy, rd_n(&rd), FALSE, FALSE);

		if (xylist_write_header(xyls)) {
			fprintf(stderr, "Failed to write xyls field header.\n");
			exit(-1);
		}

		for (j=0; j<rd_n(&rd); j++) {
			double x, y, ra, dec;
            ra  = rd_getra (&rd, j);
            dec = rd_getdec(&rd, j);
			if (!sip_radec2pixelxy(&sip, ra, dec, &x, &y)) {
				fprintf(stderr, "Point RA,Dec = (%g,%g) projects to the opposite side of the sphere.\n",
						ra, dec);
				continue;
			}
            xy_set(&xy, j, x, y);
		}
        if (xylist_write_field(xyls, &xy)) {
            fprintf(stderr, "Failed to write xyls field.\n");
            exit(-1);
        }
		if (xylist_fix_header(xyls)) {
			fprintf(stderr, "Failed to fix xyls field header.\n");
			exit(-1);
		}
        xylist_next_field(xyls);

        xy_free_data(&xy);
        rd_free_data(&rd);
	}

	if (xylist_fix_primary_header(xyls) ||
		xylist_close(xyls)) {
		fprintf(stderr, "Failed to fix header of XYLS file.\n");
		exit(-1);
	}

	if (rdlist_close(rdls)) {
		fprintf(stderr, "Failed to close RDLS file.\n");
	}

	if (verbose)
        printf("Done!\n");

	return 0;
}
