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
   \file Applies a star kdtree permutation array to the corresponding
   .id file, and also rewrites the .quad file to produce new .id, .quad
   and .skdt files that are consistent and don't require permutation.

   In:  .quad, .id, .skdt
   Out: .quad, .id, .skdt

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
#include "idfile.h"
#include "fitsioutils.h"
#include "qfits.h"
#include "starkd.h"
#include "boilerplate.h"

#define OPTIONS "hf:o:q:s"

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "    -f <input-basename>\n"
		   "   [-q <input-quadfile-basename>]\n"
		   "   [-s]: store sweep number in output star kdtree file.\n"
		   "    -o <output-basename>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **args) {
    int argchar;
    quadfile* qfin;
	quadfile* qfout;
	idfile* idin;
	idfile* idout = NULL;
	startree* treein;
	startree* treeout;
	char* progname = args[0];
	char* basein = NULL;
	char* baseout = NULL;
	char* basequadin = NULL;
	char* fn;
	int i;
	int N;
	int healpix;
	int starhp;
	int lastgrass;
	bool dosweeps = FALSE;

	qfits_header* qouthdr;
	qfits_header* qinhdr;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'q':
			basequadin = optarg;
			break;
        case 'f':
			basein = optarg;
			break;
        case 'o':
			baseout = optarg;
			break;
		case 's':
			dosweeps = TRUE;
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

	fn = mk_streefn(basein);
	printf("Reading star tree from %s ...\n", fn);
	treein = startree_open(fn);
	if (!treein) {
		fprintf(stderr, "Failed to read star kdtree from %s.\n", fn);
		exit(-1);
	}
	free_fn(fn);

	N = startree_N(treein);

	if (basequadin)
		fn = mk_quadfn(basequadin);
	else
		fn = mk_quadfn(basein);
	printf("Reading quadfile from %s ...\n", fn);
	qfin = quadfile_open(fn);
	if (!qfin) {
		fprintf(stderr, "Failed to read quadfile from %s.\n", fn);
		exit(-1);
	}
	free_fn(fn);

	fn = mk_idfn(basein);
	printf("Reading id file from %s ...\n", fn);
	idin = idfile_open(fn);
	if (!idin) {
		fprintf(stderr, "Failed to read id file from %s.  Will not generate output id file.\n", fn);
	}
	free_fn(fn);

	if (idin) {
		healpix = idin->healpix;
		if (qfin->healpix != healpix) {
			fprintf(stderr, "Quadfile header says it's healpix %i, but idfile say %i.\n",
					qfin->healpix, idin->healpix);
		}
	} else
		healpix = qfin->healpix;

	starhp = qfits_header_getint(startree_header(treein), "HEALPIX", -1);
	if (starhp == -1)
		fprintf(stderr, "Warning, input star kdtree didn't have a HEALPIX header.\n");
	else if (starhp != healpix) {
		fprintf(stderr, "Idfile says it's healpix %i, but star kdtree says %i.\n",
				healpix, starhp);
		exit(-1);
	}

	fn = mk_quadfn(baseout);
	printf("Writing quadfile to %s ...\n", fn);
	qfout = quadfile_open_for_writing(fn);
	if (!qfout) {
		fprintf(stderr, "Failed to write quadfile to %s.\n", fn);
		exit(-1);
	}
	free_fn(fn);

	if (idin) {
		fn = mk_idfn(baseout);
		printf("Writing id to %s ...\n", fn);
		idout = idfile_open_for_writing(fn);
		if (!idout) {
			fprintf(stderr, "Failed to write id file to %s.\n", fn);
			exit(-1);
		}
		free_fn(fn);
	}

	qfout->healpix = healpix;
	if (idin)
		idout->healpix = healpix;

	qfout->numstars          = qfin->numstars;
	qfout->dimquads          = qfin->dimquads;
	qfout->index_scale_upper = qfin->index_scale_upper;
	qfout->index_scale_lower = qfin->index_scale_lower;
	qfout->indexid           = qfin->indexid;

	qouthdr = quadfile_get_header(qfout);
	qinhdr  = quadfile_get_header(qfin);

	boilerplate_add_fits_headers(qouthdr);
	qfits_header_add(qouthdr, "HISTORY", "This file was created by the program \"unpermute-stars\".", NULL, NULL);
	qfits_header_add(qouthdr, "HISTORY", "unpermute-stars command line:", NULL, NULL);
	fits_add_args(qouthdr, args, argc);
	qfits_header_add(qouthdr, "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
	qfits_header_add(qouthdr, "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
	fits_copy_all_headers(qinhdr, qouthdr, "HISTORY");
	qfits_header_add(qouthdr, "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
	qfits_header_add(qouthdr, "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
	fits_copy_all_headers(qinhdr, qouthdr, "COMMENT");
	qfits_header_add(qouthdr, "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);

	if (idin) {
        qfits_header* idinhdr;
        qfits_header* idouthdr;
        idouthdr = idfile_get_header(idout);
        idinhdr = idfile_get_header(idin);
		boilerplate_add_fits_headers(idouthdr);
		qfits_header_add(idouthdr, "HISTORY", "This file was created by the program \"unpermute-stars\".", NULL, NULL);
		qfits_header_add(idouthdr, "HISTORY", "unpermute-stars command line:", NULL, NULL);
		fits_add_args(idouthdr, args, argc);
		qfits_header_add(idouthdr, "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
		qfits_header_add(idouthdr, "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
		fits_copy_all_headers(idinhdr, idouthdr, "HISTORY");
		qfits_header_add(idouthdr, "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
		qfits_header_add(idouthdr, "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
		fits_copy_all_headers(idinhdr, idouthdr, "COMMENT");
		qfits_header_add(idouthdr, "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);
	}

	if (quadfile_write_header(qfout) ||
		(idin && idfile_write_header(idout))) {
		fprintf(stderr, "Failed to write quadfile or idfile header.\n");
		exit(-1);
	}

	if (idin) {
		printf("Writing IDs...\n");
		lastgrass = 0;
		for (i=0; i<N; i++) {
			uint64_t id;
			int ind;
			if (i*80/N != lastgrass) {
				printf(".");
				fflush(stdout);
				lastgrass = i*80/N;
			}
			ind = treein->tree->perm[i];
			id = idfile_get_anid(idin, ind);
			if (idfile_write_anid(idout, id)) {
				fprintf(stderr, "Failed to write idfile entry.\n");
				exit(-1);
			}
		}
		printf("\n");
	}

	printf("Writing quads...\n");

	startree_compute_inverse_perm(treein);

	lastgrass = 0;
	for (i=0; i<qfin->numquads; i++) {
		int j;
		uint stars[qfin->dimquads];
		if (i*80/qfin->numquads != lastgrass) {
			printf(".");
			fflush(stdout);
			lastgrass = i*80/qfin->numquads;
		}
		if (quadfile_get_stars(qfin, i, stars)) {
			fprintf(stderr, "Failed to read quadfile entry.\n");
			exit(-1);
        }
		for (j=0; j<qfin->dimquads; j++)
			stars[j] = treein->inverse_perm[stars[j]];
		if (quadfile_write_quad(qfout, stars)) {
			fprintf(stderr, "Failed to write quadfile entry.\n");
			exit(-1);
		}
	}
	printf("\n");

	if (quadfile_fix_header(qfout) ||
		quadfile_close(qfout) ||
		(idin && (idfile_fix_header(idout) ||
				  idfile_close(idout)))) {
		fprintf(stderr, "Failed to close output quadfile or idfile.\n");
		exit(-1);
	}

	treeout = startree_new();
	treeout->tree = malloc(sizeof(kdtree_t));
	memcpy(treeout->tree, treein->tree, sizeof(kdtree_t));
	treeout->tree->perm = NULL;

	fits_copy_header(startree_header(treein), startree_header(treeout), "HEALPIX");
	fits_copy_header(startree_header(treein), startree_header(treeout), "ALLSKY");
	fits_copy_header(startree_header(treein), startree_header(treeout), "JITTER");
	qfits_header_add(startree_header(treeout), "HISTORY", "unpermute-stars command line:", NULL, NULL);
	fits_add_args(startree_header(treeout), args, argc);
	qfits_header_add(startree_header(treeout), "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
	qfits_header_add(startree_header(treeout), "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
	fits_copy_all_headers(startree_header(treein), startree_header(treeout), "HISTORY");
	qfits_header_add(startree_header(treeout), "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
	qfits_header_add(startree_header(treeout), "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
	fits_copy_all_headers(startree_header(treein), startree_header(treeout), "COMMENT");
	qfits_header_add(startree_header(treeout), "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);

	quadfile_close(qfin);
	if (idin)
		idfile_close(idin);

	if (dosweeps) {
		// copy sweepX headers.
		for (i=1;; i++) {
			char key[16];
			int n;
			sprintf(key, "SWEEP%i", i);
			n = qfits_header_getint(treein->header, key, -1);
			if (n == -1)
				break;
			fits_copy_header(treein->header, treeout->header, key);
		}

		// compute sweep array.
		treeout->sweep = malloc(N * sizeof(uint8_t));
		for (i=0; i<N; i++) {
			int ind = treein->tree->perm[i];
			// Stars are sorted first by sweep and then by brightness within
			// the sweep.  Instead of just storing the sweep number, we can
			// store a quantization of the total-ordered rank.
			treeout->sweep[i] = (uint8_t)(256 * ind / N);
		}
	}

	fn = mk_streefn(baseout);
	printf("Writing star kdtree to %s ...\n", fn);
	if (startree_write_to_file(treeout, fn)) {
		fprintf(stderr, "Failed to write star kdtree.\n");
		exit(-1);
	}
	free_fn(fn);
	startree_close(treein);
	free(treeout->sweep);
	free(treeout);

	return 0;
}
