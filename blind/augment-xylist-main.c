/*
 This file is part of the Astrometry.net suite.
 Copyright 2007-2008 Dustin Lang, Keir Mierle and Sam Roweis.

 The Astrometry.net suite is free software; you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301 USA
 */

/**
 * Accepts an xylist and command-line options, and produces an augmented
 * xylist.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>
#include <getopt.h>

#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "sip_qfits.h"
#include "tabsort.h"
#include "errors.h"
#include "fits-guess-scale.h"
#include "image2xy-files.h"
#include "resort-xylist.h"
#include "qfits.h"
#include "an-opts.h"
#include "augment-xylist.h"

static void print_help(const char* progname, bl* opts) {
	printf("\nUsage: %s [options]\n", progname);
    augment_xylist_print_help(stdout);
    printf("\n\n");
}


int main(int argc, char** args) {
	int c;
	int rtn;
	int help_flag = 0;
    bl* opts;
    char* me;

    augment_xylist_t theargs;
    augment_xylist_t* axy = &theargs;

    me = find_executable(args[0], NULL);

    opts = bl_new(4, sizeof(an_option_t));
    augment_xylist_add_options(opts);

    augment_xylist_init(axy);

	while (1) {
		c = opts_getopt(opts, argc, args);
		if (c == -1)
			break;
		switch (c) {
		case 0:
            fprintf(stderr, "Unknown option '-%c'\n", optopt);
            exit(-1);
        case '?':
			break;
        case 'h':
            help_flag = 1;
            break;
        default:
            if (augment_xylist_parse_option(c, optarg, axy)) {
                exit(-1);
            }
            break;
        }
    }

	rtn = 0;
	if (optind != argc) {
		int i;
		printf("Unknown arguments:\n  ");
		for (i=optind; i<argc; i++) {
			printf("%s ", args[i]);
		}
		printf("\n");
		help_flag = 1;
		rtn = -1;
	}
	if (!axy->outfn) {
		printf("Output filename (-o / --out) is required.\n");
		help_flag = 1;
		rtn = -1;
	}
	if (!(axy->imagefn || axy->xylsfn)) {
		printf("Require either an image (-i / --image) or an XYlist (-x / --xylist) input file.\n");
		help_flag = 1;
		rtn = -1;
	}
	if (help_flag) {
		print_help(args[0], opts);
		exit(rtn);
	}
    bl_free(opts);

    rtn = augment_xylist(axy, me);

    augment_xylist_free_contents(axy);

	return rtn;
}

