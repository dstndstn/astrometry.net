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

/**
   Reads a list of codes and writes a code kdtree.

   Input: .code
   Output: .ckdt
*/
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "codetree.h"
#include "boilerplate.h"

static const char* OPTIONS = "hR:i:o:bsSt:d:";

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "    -i <input-filename>\n"
		   "    -o <output-filename>\n"
		   "   (   [-b]: build bounding boxes\n"
		   "    OR [-s]: build splitting planes   )\n"
		   "    [-t  <tree type>]:  {double,float,u32,u16}, default u16.\n"
		   "    [-d  <data type>]:  {double,float,u32,u16}, default u16.\n"
		   "    [-S]: include separate splitdim array\n"
		   "    [-R <target-leaf-node-size>]   (default 25)\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argidx, argchar;
	char* progname = argv[0];
	int Nleaf = 0;
    char* treefname = NULL;
    char* codefname = NULL;
	int datatype = KDT_DATA_NULL;
	int treetype = KDT_TREE_NULL;
	int buildopts = 0;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'R':
            Nleaf = (int)strtoul(optarg, NULL, 0);
            break;
        case 'i':
            codefname = optarg;
            break;
        case 'o':
            treefname = optarg;
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
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
			exit(-1);
        default:
            return (OPT_ERR);
        }

    if (optind < argc) {
        for (argidx = optind; argidx < argc; argidx++)
            fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
		printHelp(progname);
		exit(-1);
    }
    if (!codefname || !treefname) {
		printHelp(progname);
		exit(-1);
    }

	if (codetree_files(codefname, treefname, Nleaf, datatype, treetype,
					   buildopts, argv, argc))
		exit(-1);
	return 0;
}

