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
   \file Applies a code kdtree permutation array to the corresponding
   .quad file to produce new .quad and .ckdt files that are
   consistent and don't require permutation.

   In:  .quad, .ckdt
   Out: .quad, .ckdt

   Original author: dstn
*/
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "kdtree.h"
#include "starutil.h"
#include "fileutil.h"
#include "quadfile.h"
#include "fitsioutils.h"
#include "codekd.h"
#include "qfits.h"
#include "boilerplate.h"

#define OPTIONS "hf:o:"

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "    -f <input-basename>\n"
		   "    -o <output-basename>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **args) {
    int argchar;
    quadfile* quadin;
	quadfile* quadout;
	codetree* treein;
	codetree* treeout;
	char* progname = args[0];
	char* basein = NULL;
	char* baseout = NULL;
	char* fn;
	int i;
	qfits_header* codehdr;
	qfits_header* hdr;
	int healpix;
	int codehp;
	qfits_header* qouthdr;
	qfits_header* qinhdr;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'f':
			basein = optarg;
			break;
        case 'o':
			baseout = optarg;
			break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!basein || !baseout) {
		printHelp(progname);
		exit(-1);
	}

	fn = mk_ctreefn(basein);
	printf("Reading code tree from %s ...\n", fn);
	treein = codetree_open(fn);
	if (!treein) {
		fprintf(stderr, "Failed to read code kdtree from %s.\n", fn);
		exit(-1);
	}
	free_fn(fn);
	codehdr = codetree_header(treein);

	fn = mk_quadfn(basein);
	printf("Reading quads from %s ...\n", fn);
	quadin = quadfile_open(fn);
	if (!quadin) {
		fprintf(stderr, "Failed to read quads from %s.\n", fn);
		exit(-1);
	}
	free_fn(fn);

	healpix = quadin->healpix;
	codehp = qfits_header_getint(codehdr, "HEALPIX", -1);
	if (codehp == -1)
		fprintf(stderr, "Warning, input code kdtree didn't have a HEALPIX header.\n");
	else if (codehp != healpix) {
		fprintf(stderr, "Quadfile says it's healpix %i, but code kdtree says %i.\n",
				healpix, codehp);
		exit(-1);
	}

	fn = mk_quadfn(baseout);
	printf("Writing quads to %s ...\n", fn);
	quadout = quadfile_open_for_writing(fn);
	if (!quadout) {
		fprintf(stderr, "Failed to write quads to %s.\n", fn);
		exit(-1);
	}
	free_fn(fn);

	quadout->healpix = healpix;
	quadout->indexid = quadin->indexid;
	quadout->numstars = quadin->numstars;
	quadout->dimquads = quadin->dimquads;
	quadout->index_scale_upper = quadin->index_scale_upper;
	quadout->index_scale_lower = quadin->index_scale_lower;

	qouthdr = quadfile_get_header(quadout);
	qinhdr  = quadfile_get_header(quadin);

	boilerplate_add_fits_headers(qouthdr);
	qfits_header_add(qouthdr, "HISTORY", "This file was created by the program \"unpermute-quads\".", NULL, NULL);
	qfits_header_add(qouthdr, "HISTORY", "unpermute-quads command line:", NULL, NULL);
	fits_add_args(qouthdr, args, argc);
	qfits_header_add(qouthdr, "HISTORY", "(end of unpermute-quads command line)", NULL, NULL);
	qfits_header_add(qouthdr, "HISTORY", "** unpermute-quads: history from input:", NULL, NULL);
	fits_copy_all_headers(qinhdr, qouthdr, "HISTORY");
	qfits_header_add(qouthdr, "HISTORY", "** unpermute-quads end of history from input.", NULL, NULL);
	qfits_header_add(qouthdr, "COMMENT", "** unpermute-quads: comments from input:", NULL, NULL);
	fits_copy_all_headers(qinhdr, qouthdr, "COMMENT");
	qfits_header_add(qouthdr, "COMMENT", "** unpermute-quads: end of comments from input.", NULL, NULL);
	fits_copy_header(qinhdr, qouthdr, "CXDX");
	fits_copy_header(qinhdr, qouthdr, "CXDXLT1");
	fits_copy_header(qinhdr, qouthdr, "CIRCLE");

	if (quadfile_write_header(quadout)) {
		fprintf(stderr, "Failed to write quadfile header.\n");
		exit(-1);
	}

	for (i=0; i<codetree_N(treein); i++) {
		unsigned int stars[quadin->dimquads];
		int ind = codetree_get_permuted(treein, i);
		if (quadfile_get_stars(quadin, ind, stars)) {
			fprintf(stderr, "Failed to read quad entry.\n");
			exit(-1);
        }
		if (quadfile_write_quad(quadout, stars)) {
			fprintf(stderr, "Failed to write quad entry.\n");
			exit(-1);
		}
	}

	if (quadfile_fix_header(quadout) ||
		quadfile_close(quadout)) {
		fprintf(stderr, "Failed to close output quadfile.\n");
		exit(-1);
	}

	treeout = codetree_new();
	treeout->tree = malloc(sizeof(kdtree_t));
	memcpy(treeout->tree, treein->tree, sizeof(kdtree_t));
	treeout->tree->perm = NULL;

	hdr = codetree_header(treeout);
	fits_copy_header(qinhdr, hdr, "HEALPIX");
	fits_copy_header(qinhdr, hdr, "HPNSIDE");
	boilerplate_add_fits_headers(hdr);
	qfits_header_add(hdr, "HISTORY", "This file was created by the program \"unpermute-quads\".", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "unpermute-quads command line:", NULL, NULL);
	fits_add_args(hdr, args, argc);
	qfits_header_add(hdr, "HISTORY", "(end of unpermute-quads command line)", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "** unpermute-quads: history from input ckdt:", NULL, NULL);
	fits_copy_all_headers(codehdr, hdr, "HISTORY");
	qfits_header_add(hdr, "HISTORY", "** unpermute-quads end of history from input ckdt.", NULL, NULL);
	qfits_header_add(hdr, "COMMENT", "** unpermute-quads: comments from input ckdt:", NULL, NULL);
	fits_copy_all_headers(codehdr, hdr, "COMMENT");
	qfits_header_add(hdr, "COMMENT", "** unpermute-quads: end of comments from input ckdt.", NULL, NULL);
	fits_copy_header(codehdr, hdr, "CXDX");
	fits_copy_header(codehdr, hdr, "CXDXLT1");
	fits_copy_header(codehdr, hdr, "CIRCLE");

	quadfile_close(quadin);

	fn = mk_ctreefn(baseout);
	printf("Writing code kdtree to %s ...\n", fn);
	if (codetree_write_to_file(treeout, fn) ||
		codetree_close(treeout)) {
		fprintf(stderr, "Failed to write code kdtree.\n");
		exit(-1);
	}
	free_fn(fn);

	codetree_close(treein);
	free(treeout);

	printf("Done!\n");

	return 0;
}
