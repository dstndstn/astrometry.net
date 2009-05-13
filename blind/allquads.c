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

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "fitsioutils.h"
#include "qfits.h"
#include "starkd.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"

#define OPTIONS "hi:c:q:u:l:d:I:v"

static void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
	       "      -i <input-filename>    (star kdtree (skdt.fits) input file)\n"
		   "      -c <codes-output-filename>    (codes file (code.fits) output file)\n"
           "      -q <quads-output-filename>    (quads file (quad.fits) output file)\n"
	       "     [-u <scale>]    upper bound of quad scale (arcmin)\n"
	       "     [-l <scale>]    lower bound of quad scale (arcmin)\n"
		   "     [-d <dimquads>] number of stars in a \"quad\".\n"
		   "     [-I <unique-id>] set the unique ID of this index\n\n"
		   "\nReads skdt, writes {code, quad}.\n\n"
	       , progname);
}

extern char *optarg;
extern int optind, opterr, optopt;


static void add_interior_stars(unsigned int* quad, int starnum, int firstindex,
							   il* starsC, int dimquads, int dimcodes,
							   startree_t* starkd, codefile* codes, quadfile* quads) {
	int i;
	for (i=firstindex; i<il_size(starsC); i++) {
		quad[starnum] = il_get(starsC, i);
		// Did we just add the last star?
		if (starnum == dimquads-1) {
			if (log_get_level() >= LOG_VERB) {
				int k;
				logverb("  quad: ");
				for (k=0; k<dimquads; k++)
					logverb("%-6i ", quad[k]);
				logverb("\n");
			}
			quad_write_const(codes, quads, quad, starkd, dimquads, dimcodes);
		} else {
			// Recurse.
			add_interior_stars(quad, starnum+1, i+1, starsC, dimquads,
							   dimcodes, starkd, codes, quads);
		}
	}
}


static void build_all_quads(int starA, int starB, il* starsC,
							startree_t* starkd, int dimquads, int dimcodes,
							codefile* codes, quadfile* quads) {
	unsigned int quad[DQMAX];
	quad[0] = starA;
	quad[1] = starB;
	add_interior_stars(quad, 2, 0, starsC, dimquads, dimcodes, starkd, codes, quads);
}

int main(int argc, char** argv) {
	int argchar;

	startree_t* starkd;
	quadfile* quads;
	codefile* codes;

	double quad_dist2_upper = HUGE_VAL;
	double quad_dist2_lower = 0.0;

	char *quadfn = NULL;
	char *codefn = NULL;
	char *skdtfn = NULL;

	qfits_header* qhdr;
	qfits_header* chdr;

	int dimquads = 4;
	int dimcodes;
	int loglvl = LOG_MSG;

	int id = 0;
	int i, j, k, N;
	int lastgrass = 0;
	int hp, hpnside;

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'v':
			loglvl++;
			break;
		case 'd':
			dimquads = atoi(optarg);
			break;
		case 'I':
			id = atoi(optarg);
			break;
		case 'h':
			print_help(argv[0]);
			exit(0);
		case 'i':
            skdtfn = optarg;
			break;
		case 'c':
            codefn = optarg;
            break;
        case 'q':
            quadfn = optarg;
            break;
		case 'u':
			quad_dist2_upper = arcmin2distsq(atof(optarg));
			break;
		case 'l':
			quad_dist2_lower = arcmin2distsq(atof(optarg));
			break;
		default:
			return -1;
		}

	log_init(loglvl);

	if (!skdtfn || !codefn || !quadfn) {
		printf("Specify in & out filenames, bonehead!\n");
		print_help(argv[0]);
		exit( -1);
	}

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

	if (!id)
		logmsg("Warning: you should set the unique-id for this index (-i).\n");

	if (dimquads > DQMAX) {
		ERROR("Quad dimension %i exceeds compiled-in max %i.\n", dimquads, DQMAX);
		exit(-1);
	}
	dimcodes = dimquad2dimcode(dimquads);

	printf("Reading star kdtree %s ...\n", skdtfn);
	starkd = startree_open(skdtfn);
	if (!starkd) {
		ERROR("Failed to open star kdtree %s\n", skdtfn);
		exit( -1);
	}
	printf("Star tree contains %i objects.\n", startree_N(starkd));

	printf("Will write to quad file %s and code file %s\n", quadfn, codefn);
    quads = quadfile_open_for_writing(quadfn);
	if (!quads) {
		ERROR("Couldn't open file %s to write quads.\n", quadfn);
		exit(-1);
	}
    codes = codefile_open_for_writing(codefn);
	if (!codes) {
		ERROR("Couldn't open file %s to write codes.\n", quadfn);
		exit(-1);
	}

	quads->dimquads = dimquads;
	codes->dimcodes = dimcodes;

	if (id) {
		quads->indexid = id;
		codes->indexid = id;
	}

	// get the "HEALPIX" header from the skdt and put it in the code and quad headers.
	hp = qfits_header_getint(startree_header(starkd), "HEALPIX", -1);
	if (hp == -1) {
		logmsg("Warning: skdt does not contain \"HEALPIX\" header.  Code and quad files will not contain this header either.\n");
	}
	quads->healpix = hp;
	codes->healpix = hp;
    // likewise "HPNSIDE"
	hpnside = qfits_header_getint(startree_header(starkd), "HPNSIDE", 1);
	quads->hpnside = hpnside;
	codes->hpnside = hpnside;

	qhdr = quadfile_get_header(quads);
	chdr = codefile_get_header(codes);

	qfits_header_add(qhdr, "CXDX", "T", "All codes have the property cx<=dx.", NULL);
	qfits_header_add(qhdr, "CXDXLT1", "T", "All codes have the property cx+dx<=1.", NULL);
	qfits_header_add(qhdr, "MIDHALF", "T", "All codes have the property cx+dx<=1.", NULL);
	qfits_header_add(qhdr, "CIRCLE", "T", "Codes live in the circle, not the box.", NULL);

	qfits_header_add(chdr, "CXDX", "T", "All codes have the property cx<=dx.", NULL);
	qfits_header_add(chdr, "CXDXLT1", "T", "All codes have the property cx+dx<=1.", NULL);
	qfits_header_add(chdr, "MIDHALF", "T", "All codes have the property cx+dx<=1.", NULL);
	qfits_header_add(chdr, "CIRCLE", "T", "Codes live in the circle, not the box.", NULL);

    if (quadfile_write_header(quads)) {
        ERROR("Couldn't write headers to quads file %s\n", quadfn);
        exit(-1);
    }
    if (codefile_write_header(codes)) {
        ERROR("Couldn't write headers to code file %s\n", codefn);
        exit(-1);
    }

    codes->numstars = startree_N(starkd);
    codes->index_scale_upper = distsq2rad(quad_dist2_upper);
    codes->index_scale_lower = distsq2rad(quad_dist2_lower);

	quads->numstars = codes->numstars;
    quads->index_scale_upper = codes->index_scale_upper;
    quads->index_scale_lower = codes->index_scale_lower;


	N = startree_N(starkd);
	// star A = i
	for (i=0; i<N; i++) {
		double xyzA[3];
		int* inds;
		int NR;
		int nq;

		int grass = (i*80 / N);
		if (grass != lastgrass) {
			printf(".");
			fflush(stdout);
			lastgrass = grass;
		}

		startree_get(starkd, i, xyzA);
		startree_search_for(starkd, xyzA, quad_dist2_upper,
							NULL, NULL, &inds, &NR);

		nq = quads->numquads;

		// star B = inds[j]
		for (j=0; j<NR; j++) {
			double xyzB[3];
			double mid[3];
			double qr2;
			il* indsC;

			if (inds[j] <= i)
				continue;

			startree_get(starkd, inds[j], xyzB);
			qr2 = distsq(xyzA, xyzB, 3);
			if (qr2 < quad_dist2_lower)
				continue;
			assert(qr2 < quad_dist2_upper);

			// quad center
			star_midpoint(mid, xyzA, xyzB);
			// quad diameter -> radius
			qr2 /= 4;

			indsC = il_new(32);
			// stars C = inds[k]: subset of inds that are inside the quad circle.
			for (k=0; k<NR; k++) {
				double xyzC[3];
				double d2;
				if (k == j)
					continue;
				if (inds[k] == i)
					continue;
				startree_get(starkd, inds[k], xyzC);
				d2 = distsq(mid, xyzC, 3);
				if (d2 > qr2)
					continue;
				il_append(indsC, inds[k]);
			}

			build_all_quads(i, inds[j], indsC, starkd, dimquads, dimcodes,
							codes, quads);

			il_free(indsC);
		}

		logverb("Star %i of %i: wrote %i quads for this star, total %i so far.\n",
				i+1, N, quads->numquads - nq, quads->numquads);
	}

	startree_close(starkd);

	// fix output file headers.
	if (quadfile_fix_header(quads) ||
		quadfile_close(quads)) {
		ERROR("Couldn't write quad output file: %s\n", strerror(errno));
		exit( -1);
	}
	if (codefile_fix_header(codes) ||
		codefile_close(codes)) {
		ERROR("Couldn't write code output file: %s\n", strerror(errno));
		exit( -1);
	}

	printf("Done.\n");
	return 0;
}

