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
#include <errno.h>
#include <string.h>

#include "an-bool.h"
#include "bl.h"
#include "boilerplate.h"
#include "wcs-rd2xy.h"
#include "errors.h"

const char* OPTIONS = "hi:o:w:f:R:D:t";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   -i <rdls input file>\n"
		   "   -o <xyls output file>\n"
		   "  [-f <rdls field index>] (default: all)\n"
		   "  [-R <RA-column-name> -D <Dec-column-name>]\n"
		   "  [-t]: just use TAN projection, even if SIP extension exists.\n"
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
	il* fields;

	fields = il_new(16);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
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

	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}

	if (!xylsfn || !rdlsfn || !wcsfn) {
		print_help(args[0]);
		exit(-1);
	}

    if (wcs_rd2xy(wcsfn, rdlsfn, xylsfn,
                  rcol, dcol, forcetan, fields)) {
        ERROR("wcs-rd2xy failed");
        exit(-1);
    }

	return 0;
}
