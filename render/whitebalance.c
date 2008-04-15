/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, Dustin Lang, Keir Mierle and Sam Roweis.

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

/*
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#include "pnm.h"

#include "pnmutils.h"

static const char* OPTIONS = "h";

static void printHelp(char* progname) {
	fprintf(stderr, "usage: %s\n"
			"    <input-image-1>  <input-image-2>\n"
			"Writes gain values that yield the nearest match between the images.\n"
			"\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int
main(int argc, char** args) {
	char* progname = args[0];
	int argchar;
	char* fn1;
	char* fn2;
	double redgain, bluegain, greengain;

    pnm_init(&argc, args);

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'h':
			printHelp(progname);
			exit(0);
		}

	if (optind != argc - 2) {
		printHelp(progname);
		exit(-1);
	}
	fn1 = args[optind];
	fn2 = args[optind+1];

	if (pnmutils_whitebalance(fn1, fn2, &redgain, &greengain, &bluegain)) {
		exit(-1);
	}

	printf("R %g, G %g, B %g\n", redgain, greengain, bluegain);

    return 0;
}
