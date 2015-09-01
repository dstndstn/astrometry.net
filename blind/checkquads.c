/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

/*
  Checks the consistency of "quad" and "qidx" files.
*/

#include <string.h>

#include "starutil.h"
#include "quadfile.h"
#include "qidxfile.h"
#include "bl.h"
#include "fitsioutils.h"

#define OPTIONS "hf:"
const char HelpString[] =
"quadidx -f fname\n";


int main(int argc, char *argv[]) {
	int argidx, argchar;
	char *qidxfname = NULL;
	char *quadfname = NULL;
	quadfile* quad;
	qidxfile* qidx;
	int q, s;
	int dimquads;
	
	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'f':
			qidxfname = mk_qidxfn(optarg);
			quadfname = mk_quadfn(optarg);
			break;
		case '?':
			fprintf(stderr, "Unknown option `-%c'.\n", optopt);
		case 'h':
			fprintf(stderr, HelpString);
			return (HELP_ERR);
		default:
			return (OPT_ERR);
		}

	if (optind < argc) {
		for (argidx = optind; argidx < argc; argidx++)
			fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
		fprintf(stderr, HelpString);
		return (OPT_ERR);
	}

	quad = quadfile_open(quadfname, 0);
	if (!quad) {
		fprintf(stderr, "Couldn't open quads file %s.\n", quadfname);
		exit(-1);
	}

	qidx = qidxfile_open(qidxfname, 0);
	if (!qidx) {
		fprintf(stderr, "Couldn't open qidx file %s.\n", qidxfname);
		exit(-1);
	}

	if (quad->numquads != qidx->numquads) {
		fprintf(stderr, "Number of quads does not agree: %i vs %i\n",
				quad->numquads, qidx->numquads);
		exit(-1);
	}

	dimquads = quadfile_dimquads(quad);

	printf("Checking stars...\n");
	for (s=0; s<qidx->numstars; s++) {
		uint32_t* quads;
		int nquads;
		int j;
		qidxfile_get_quads(qidx, s, &quads, &nquads);
		for (j=0; j<nquads; j++) {
			int star[dimquads];
			int k, n;
			quadfile_get_stars(quad, quads[j], star);
			n = 0;
			for (k=0; k<dimquads; k++) {
				if (star[k] == s)
					n++;
			}
			if (n != 1) {
				fprintf(stderr, "Star %i, quad %i: found %i instances of the quad in the qidx (not 1)\n",
						s, quads[j], n);
				fprintf(stderr, "  found: ");
				for (k=0; k<dimquads; k++) {
					fprintf(stderr, "%i ", star[k]);
				}
				fprintf(stderr, "\n");
			}
		}
	}

	printf("Checking quads...\n");
	for (q=0; q<quad->numquads; q++) {
		int star[dimquads];
		uint32_t* quads;
		int nquads;
		int j;
		quadfile_get_stars(quad, q, star);
		for (j=0; j<dimquads; j++) {
			int k;
			int n;
			qidxfile_get_quads(qidx, star[j], &quads, &nquads);
			n = 0;
			for (k=0; k<nquads; k++) {
				if (quads[k] == q)
					n++;
			}
			if (n != 1) {
				fprintf(stderr, "Quad %i, star %i: found %i instances of the quad in the qidx (not 1)\n",
						q, star[j], n);
				fprintf(stderr, "  found: ");
				for (k=0; k<nquads; k++) {
					fprintf(stderr, "%i ", quads[k]);
				}
				fprintf(stderr, "\n");
			}
		}
	}

	quadfile_close(quad);
	qidxfile_close(qidx);

	return 0;
}
