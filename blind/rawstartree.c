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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "starutil.h"
#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "an-catalog.h"
#include "mathutil.h"

#define OPTIONS "ho:l:m:"

/* make a raw startree from a catalog; no cutting */

void printHelp(char* progname)
{
	fprintf(stderr, "%s usage:\n"
	        "   -o <output-file>\n"
	        "  [-l <n-leaf-point>]: Target number of points in the leaves of the tree (default 15).\n"
	        "\n"
	        "  <input-catalog> ...\n"
	        "\n"
	        "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args)
{
	int argchar;
	char* progname = args[0];
	char* outfn = NULL;
	int i;
	int Nleaf = 15;
	time_t start;
	int N;
	int infile;
	double* xyz;
	kdtree_t* kd;

	start = time(NULL);

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
		case 'h':
			printHelp(progname);
			exit(0);
		case 'o':
			outfn = optarg;
			break;
		case 'l':
			Nleaf = atoi(optarg);
			break;
		}

	if (!outfn || (optind == argc)) {
		printHelp(progname);
		exit( -1);
	}

	N = 0;
	for (infile = optind; infile < argc; infile++) {
		char* fn;
		an_catalog* ancat;
		fn = args[infile];
		ancat = an_catalog_open(fn);
		if (!ancat) {
			fprintf(stderr, "Failed to open Astrometry.net catalog %s\n", fn);
			exit( -1);
		}
		N += ancat->nentries;
		an_catalog_close(ancat);
	}

	xyz = malloc(N * 3 * sizeof(double));

	if (!xyz) {
		fprintf(stderr, "Failed to allocate arrays for star positions.\n");
		exit( -1);
	}

	i = 0;
	for (infile = optind; infile < argc; infile++) {
		char* fn;
		int j;
		an_catalog* ancat;
		int lastgrass = 0;

		fn = args[infile];
		ancat = an_catalog_open(fn);
		if (!ancat) {
			fprintf(stderr, "Failed to open Astrometry.net catalog %s\n", fn);
			exit( -1);
		}

		fprintf(stderr, "Reading %i entries for catalog file %s.\n", ancat->nentries, fn);

		for (j = 0; j < ancat->nentries; j++, i++) {
			int grass;
			an_entry* entry;

			entry = an_catalog_read_entry(ancat);
			if (!entry)
				break;

			grass = (j * 80 / ancat->nentries);
			if (grass != lastgrass) {
				fprintf(stderr, ".");
				fflush(stderr);
				lastgrass = grass;
			}

			radec2xyzarr(deg2rad(entry->ra), deg2rad(entry->dec), xyz+3*i);
		}
		an_catalog_close(ancat);
	}
	fprintf(stderr, "\n");

	fprintf(stderr, "Creating kdtree...\n");
	kd = kdtree_build(NULL, xyz, N, 3, Nleaf, KDTT_DOUBLE,
					  KD_BUILD_BBOX | KD_BUILD_SPLIT | KD_BUILD_SPLITDIM);
	if (!kd) {
		fprintf(stderr, "Failed to build kdtree.\n");
		exit( -1);
	}

	fprintf(stderr, "Built kdtree with %i nodes\n", kd->nnodes);

	fprintf(stderr, "Writing output to file %s...\n", outfn);
	if (kdtree_fits_write(kd, outfn, NULL)) { // FIXME add extra headers?
		fprintf(stderr, "Failed to write kdtree to file %s.\n", outfn);
		exit( -1);
	}

	kdtree_free(kd);

	return 0;
}

