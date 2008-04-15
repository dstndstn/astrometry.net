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

/**
   Builds a star kdtree from a list of stars.

   Input: .objs
   Output: .skdt
*/
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "starutil.h"
#include "fileutil.h"
#include "catalog.h"
#include "fitsioutils.h"
#include "starkd.h"
#include "boilerplate.h"

#define OPTIONS "hR:f:k:d:t:bsSc"

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "     -f <input-catalog-name>\n"
		   "   (   [-b]: build bounding boxes\n"
		   "    OR [-s]: build splitting planes   )\n"
		   "    [-R Nleaf]: number of points in a kdtree leaf node (default 25)\n"
		   "    [-k keep]:  number of points to keep\n"
		   "    [-t  <tree type>]:  {double,float,u32,u16}, default u32.\n"
		   "    [-d  <data type>]:  {double,float,u32,u16}, default u32.\n"
		   "    [-S]: include separate splitdim array\n"
		   "    [-c]: run kdtree_check on the resulting tree\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argidx, argchar;
    int nkeep = 0;
	startree* starkd;
    catalog* cat;
    int Nleaf = 25;
    char* basename = NULL;
    char* treefname = NULL;
    char* catfname = NULL;
	char* progname = argv[0];

	int exttype  = KDT_EXT_DOUBLE;
	int datatype = KDT_DATA_NULL;
	int treetype = KDT_TREE_NULL;
	bool convert = FALSE;
	int tt;
	int buildopts = 0;
	int i, N, D;
	int checktree = 0;

	qfits_header* catheader = NULL;

    if (argc <= 2) {
		printHelp(progname);
        return 0;
    }

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'c':
			checktree = 1;
			break;
        case 'R':
            Nleaf = (int)strtoul(optarg, NULL, 0);
            break;
        case 'k':
            nkeep = atoi(optarg);
            if (nkeep == 0) {
                printf("Couldn't parse \'keep\': \"%s\"\n", optarg);
                exit(-1);
            }
            break;
        case 'f':
            basename = optarg;
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

	if (!(buildopts & (KD_BUILD_BBOX | KD_BUILD_SPLIT))) {
		printf("You need bounding-boxes or splitting planes!\n");
		printHelp(progname);
		exit(-1);
	}

    if (!basename) {
        printHelp(progname);
        exit(-1);
    }

	// defaults
	if (!datatype)
		datatype = KDT_DATA_U32;
	if (!treetype)
		treetype = KDT_TREE_U32;

	// the outside world works in doubles.
	if (datatype != KDT_DATA_DOUBLE)
		convert = TRUE;

    catfname = mk_catfn(basename);
	treefname = mk_streefn(basename);
    fprintf(stderr, "%s: building KD tree for %s\n", argv[0], catfname);
	fprintf(stderr, "Will write output to %s\n", treefname);

    fprintf(stderr, "Reading star catalogue...");
    cat = catalog_open(catfname);
    free_fn(catfname);
    if (!cat) {
        fprintf(stderr, "Couldn't read catalogue.\n");
        exit(-1);
    }
    fprintf(stderr, "got %i stars.\n", cat->numstars);

    if (nkeep && (nkeep < cat->numstars)) {
        cat->numstars = nkeep;
        fprintf(stderr, "keeping at most %i stars.\n", nkeep);
    }

	starkd = startree_new();
	if (!starkd) {
		fprintf(stderr, "Failed to allocate startree.\n");
		exit(-1);
	}

	tt = kdtree_kdtypes_to_treetype(exttype, treetype, datatype);
	N = cat->numstars;
	D = DIM_STARS;
	starkd->tree = kdtree_new(N, D, Nleaf);
	{
		double low[D];
		double high[D];
		int d;
		for (d=0; d<D; d++) {
			low[d] = -1.0;
			high[d] = 1.0;
		}
		kdtree_set_limits(starkd->tree, low, high);
	}
	if (convert) {
		fprintf(stderr, "Converting data...\n");
		fflush(stderr);
		starkd->tree = kdtree_convert_data(starkd->tree, catalog_get_base(cat),
										   N, D, Nleaf, tt);
		// close the catalog to save space, but grab the FITS header first.
		catheader = qfits_header_copy(catalog_get_header(cat));
        catalog_close(cat);

		fprintf(stderr, "Building tree...\n");
		fflush(stderr);
		starkd->tree = kdtree_build(starkd->tree, starkd->tree->data.any, N, D,
									Nleaf, tt, buildopts);
	} else {
		fprintf(stderr, "Building tree...\n");
		fflush(stderr);
		starkd->tree = kdtree_build(NULL, catalog_get_base(cat), N, D,
									Nleaf, tt, buildopts);
		catheader = qfits_header_copy(catalog_get_header(cat));
        catalog_close(cat);
	}

    if (!starkd->tree) {
        fprintf(stderr, "Couldn't build kdtree.\n");
        exit(-1);
    }

	if (checktree) {
		fprintf(stderr, "Checking tree...\n");
		if (kdtree_check(starkd->tree)) {
			fprintf(stderr, "\n\nTree check failed!!\n\n\n");
		}
	}

	fprintf(stderr, "Writing output to %s ...\n", treefname);
	fflush(stderr);
	fits_header_add_int(startree_header(starkd), "NLEAF", Nleaf, "Target number of points in leaves.");
	fits_header_add_int(startree_header(starkd), "KEEP", nkeep, "Number of stars kept (0=no limit).");
	fits_copy_header(catheader, startree_header(starkd), "HEALPIX");
	fits_copy_header(catheader, startree_header(starkd), "ALLSKY");
	fits_copy_header(catheader, startree_header(starkd), "JITTER");
	boilerplate_add_fits_headers(startree_header(starkd));
	qfits_header_add(startree_header(starkd), "HISTORY", "This file was created by the program \"startree\".", NULL, NULL);
	qfits_header_add(startree_header(starkd), "HISTORY", "startree command line:", NULL, NULL);
	fits_add_args(startree_header(starkd), argv, argc);
	qfits_header_add(startree_header(starkd), "HISTORY", "(end of startree command line)", NULL, NULL);
	qfits_header_add(startree_header(starkd), "HISTORY", "** History entries copied from the input file:", NULL, NULL);
	fits_copy_all_headers(catheader, startree_header(starkd), "HISTORY");
	qfits_header_add(startree_header(starkd), "HISTORY", "** End of history entries.", NULL, NULL);

	for (i=1;; i++) {
		char key[16];
		int n;
		sprintf(key, "SWEEP%i", i);
		n = qfits_header_getint(catheader, key, -1);
		if (n == -1)
			break;
		fits_copy_header(catheader, startree_header(starkd), key);
	}

	if (startree_write_to_file(starkd, treefname)) {
		fprintf(stderr, "Failed to write star kdtree.\n");
		exit(-1);
	}
    free_fn(treefname);
    fprintf(stderr, "done.\n");

    startree_close(starkd);
	qfits_header_destroy(catheader);
    return 0;
}

