/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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
#include <math.h>
#include <string.h>

#include "startree2.h"
#include "fitstable.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "fitsioutils.h"

const char* OPTIONS = "hvL:d:t:bsSci:o:R:D:";

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "     -i <input-fits-catalog-name>\n"
           "     -o <output-star-kdtree-name>\n"
		   "    [-R <ra-column-name>]: name of RA in FITS table (default RA)\n"
		   "    [-D <dec-column-name>]: name of DEC in FITS table (default DEC)\n"
		   "    [-b]: build bounding boxes (default: splitting planes)\n"
		   "    [-L Nleaf]: number of points in a kdtree leaf node (default 25)\n"
		   "    [-t  <tree type>]:  {double,float,u32,u16}, default u32.\n"
		   "    [-d  <data type>]:  {double,float,u32,u16}, default u32.\n"
		   "    [-S]: include separate splitdim array\n"
		   "    [-c]: run kdtree_check on the resulting tree\n"
		   "    [-v]: +verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argidx, argchar;
	startree_t* starkd;
	fitstable_t* cat;
	fitstable_t* tag;
    int Nleaf = 0;
    char* skdtfn = NULL;
    char* catfn = NULL;
	char* progname = argv[0];
	char* racol = NULL;
	char* deccol = NULL;
	int loglvl = LOG_MSG;

	int datatype = 0;
	int treetype = 0;
	int buildopts = 0;
	anbool checktree = FALSE;

    if (argc <= 2) {
		printHelp(progname);
        return 0;
    }

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'R':
			racol = optarg;
			break;
		case 'D':
			deccol = optarg;
			break;
		case 'c':
			checktree = TRUE;
			break;
        case 'L':
            Nleaf = (int)strtoul(optarg, NULL, 0);
            break;
        case 'i':
            catfn = optarg;
            break;
        case 'o':
            skdtfn = optarg;
            break;
		case 't':
			treetype = kdtree_kdtype_parse_tree_string(optarg);
			break;
		case 'd':
			datatype = kdtree_kdtype_parse_data_string(optarg);
			break;
		case 'b':
			buildopts |= KD_BUILD_BBOX;
			break;
		case 's':
			buildopts |= KD_BUILD_SPLIT;
			break;
		case 'S':
			buildopts |= KD_BUILD_SPLITDIM;
			break;
		case 'v':
			loglvl++;
			break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (optind < argc) {
        for (argidx = optind; argidx < argc; argidx++)
            fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
		printHelp(progname);
		exit(-1);
    }

    if (!(catfn && skdtfn)) {
        printHelp(progname);
        exit(-1);
    }

	log_init(loglvl);
	fits_use_error_system();

	logmsg("Building star kdtree: reading %s, writing to %s\n", catfn, skdtfn);

    logverb("Reading star catalogue...");
	cat = fitstable_open(catfn);
    if (!cat) {
        ERROR("Couldn't read catalog");
        exit(-1);
    }
	logmsg("Got %i stars\n", fitstable_nrows(cat));

	starkd = startree_build(cat, racol, deccol, datatype, treetype,
							buildopts, Nleaf, argv, argc);
	if (!starkd) {
		ERROR("Failed to create star kdtree");
		exit(-1);
	}
	if (checktree) {
		logverb("Checking tree...\n");
		if (kdtree_check(starkd->tree)) {
			ERROR("kdtree_check failed!");
			exit(-1);
		}
	}
	if (startree_write_to_file(starkd, skdtfn)) {
		ERROR("Failed to write star kdtree");
		exit(-1);
	}
    startree_close(starkd);

	// Append tag-along table.
	logmsg("Writing tag-along data...\n");
	tag = fitstable_open_for_appending(skdtfn);

	if (startree_write_tagalong_table(cat, tag, racol, deccol)) {
		ERROR("Failed to write tag-along table");
		exit(-1);
	}

	if (fitstable_close(tag)) {
		ERROR("Failed to close tag-along data");
		exit(-1);
	}
	
	fitstable_close(cat);
    return 0;
}


