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
#include <errno.h>
#include <string.h>

#include "codefile.h"
#include "fitsioutils.h"
#include "codekd.h"
#include "boilerplate.h"

#define OPTIONS "hR:i:o:bsSt:d:"

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

	int Nleaf = 25;
    codetree *codekd = NULL;
    char* treefname = NULL;
    char* codefname = NULL;
    codefile* codes;
	int rtn;
	qfits_header* hdr;
	int exttype = KDT_EXT_DOUBLE;
	int datatype = KDT_DATA_NULL;
	int treetype = KDT_TREE_NULL;
	int tt;
	int buildopts = 0;
	int N, D;
    qfits_header* chdr;

    if (argc <= 2) {
        printHelp(progname);
		exit(-1);
    }

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
	if (!(buildopts & (KD_BUILD_BBOX | KD_BUILD_SPLIT))) {
		printf("You need bounding-boxes or splitting planes!\n");
		printHelp(progname);
		exit(-1);
	}

	// defaults
	if (!datatype)
		datatype = KDT_DATA_U16;
	if (!treetype)
		treetype = KDT_TREE_U16;

    fprintf(stderr, "codetree: building KD tree for %s\n", codefname);
    fprintf(stderr, "       will write KD tree file %s\n", treefname);

    fprintf(stderr, "  Reading codes...");
    fflush(stderr);

    codes = codefile_open(codefname);
    if (!codes) {
        exit(-1);
    }
    fprintf(stderr, "got %u codes.\n", codes->numcodes);

	codekd = codetree_new();
	if (!codekd) {
		fprintf(stderr, "Failed to allocate a codetree structure.\n");
		exit(-1);
	}

	tt = kdtree_kdtypes_to_treetype(exttype, treetype, datatype);
	N = codes->numcodes;
	D = codefile_dimcodes(codes);
	codekd->tree = kdtree_new(N, D, Nleaf);

    chdr = codefile_get_header(codes);
	{
		double low[D];
		double high[D];
		int d;
		bool circ;
        circ = qfits_header_getboolean(chdr, "CIRCLE", 0);
		for (d=0; d<D; d++) {
			if (circ) {
				low [d] = 0.5 - M_SQRT1_2;
				high[d] = 0.5 + M_SQRT1_2;
			} else {
				low [d] = 0.0;
				high[d] = 1.0;
			}
		}
		kdtree_set_limits(codekd->tree, low, high);
	}
    fprintf(stderr, "Building tree...");
    fflush(stderr);
    codekd->tree = kdtree_build(codekd->tree, codes->codearray, N, D,
                                Nleaf, tt, buildopts);
    if (!codekd->tree) {
		fprintf(stderr, "Failed to build code kdtree.\n");
		exit(-1);
	}
    fprintf(stderr, "done (%d nodes)\n", codetree_N(codekd));

    codekd->tree->name = strdup(CODETREE_NAME);

    fprintf(stderr, "  Writing code KD tree to %s...", treefname);
    fflush(stderr);

	hdr = codetree_header(codekd);
	fits_header_add_int(hdr, "NLEAF", Nleaf, "Target number of points in leaves.");
	fits_copy_header(chdr, hdr, "INDEXID");
	fits_copy_header(chdr, hdr, "HEALPIX");
	fits_copy_header(chdr, hdr, "HPNSIDE");
	fits_copy_header(chdr, hdr, "CXDX");
	fits_copy_header(chdr, hdr, "CXDXLT1");
	fits_copy_header(chdr, hdr, "CIRCLE");

	boilerplate_add_fits_headers(hdr);
	qfits_header_add(hdr, "HISTORY", "This file was created by the program \"codetree\".", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "codetree command line:", NULL, NULL);
	fits_add_args(hdr, argv, argc);
	qfits_header_add(hdr, "HISTORY", "(end of codetree command line)", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "** codetree: history from input file:", NULL, NULL);
	fits_copy_all_headers(chdr, hdr, "HISTORY");
	qfits_header_add(hdr, "HISTORY", "** codetree: end of history from input file.", NULL, NULL);

	rtn = codetree_write_to_file(codekd, treefname);
	if (rtn) {
        fprintf(stderr, "Couldn't write code kdtree.\n");
        exit(-1);
    }

    fprintf(stderr, "done.\n");
    codefile_close(codes);

    kdtree_free(codekd->tree);
    codekd->tree = NULL;
    codetree_close(codekd);
	return 0;
}

