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

#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "quadfile.h"
#include "catalog.h"
#include "kdtree.h"
#include "starutil.h"
#include "bl.h"
#include "starkd.h"
#include "boilerplate.h"

#define OPTIONS "hn:o:"

extern char *optarg;
extern int optind, opterr, optopt;

void print_help(char* progname)
{
	boilerplate_help_header(stderr);
	fprintf(stderr, "Usage: %s\n"
			"   -o <output-image-file-name>\n"
			"   [-n <image-size>]  (default 3000)\n"
			"   [-h]: help\n"
			"   <base-name> [<base-name> ...]\n\n"
			"Requires both (objs or skdt) and quad files.  Writes a PGM image.\n"
			"If you specify multiple -o and -n options, multiple image files will be written.\n\n",
	        progname);
}

int main(int argc, char** args) {
    int argchar;
	const int Ndefault = 3000;
	int N = Ndefault;
	char* basename;
	char* outfn = NULL;
	char* fn;
	quadfile* qf;
	catalog* cat = NULL;
	startree_t* skdt = NULL;
	uchar* img = NULL;
	int i;
	int maxval;
	unsigned char* starcounts;
	il* imgsizes;
	pl* outnames;
	unsigned int** counts;
	int Nimgs;
    int dimquads;

	imgsizes = il_new(8);
	outnames = pl_new(8);

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
		case 'o':
			outfn = optarg;
			pl_append(outnames, strdup(optarg));
			break;
        case 'n':
            N = atoi(optarg);
			il_append(imgsizes, N);
            break;
		case 'h':
			print_help(args[0]);
			exit(0);
		}

	if (!outfn || !N || (optind == argc)) {
		print_help(args[0]);
		exit(-1);
	}

	for (i=il_size(imgsizes); i<pl_size(outnames); i++)
		il_append(imgsizes, Ndefault);

	Nimgs = pl_size(outnames);

	counts = malloc(Nimgs * sizeof(unsigned int*));
	for (i=0; i<Nimgs; i++) {
		int n = il_get(imgsizes, i);
		counts[i] = calloc(n*n, sizeof(int));
		if (!counts) {
			fprintf(stderr, "Couldn't allocate %ix%i image.\n", N, N);
			exit(-1);
		}
	}

	for (; optind<argc; optind++) {
		int Nstars;
		basename = args[optind];
		printf("Reading files with basename %s\n", basename);

        asprintf(&fn, "%s.quad.fits", basename);
		qf = quadfile_open(fn);
		if (!qf) {
			fprintf(stderr, "Failed to open quad file %s.\n", fn);
			continue;
		}
		free(fn);

        asprintf(&fn, "%s.objs.fits", basename);
		printf("Trying to open catalog file %s...\n", fn);
		cat = catalog_open(fn);
		free(fn);
		if (cat) {
			Nstars = cat->numstars;
		} else {
            asprintf(&fn, "%s.skdt.fits", basename);
			printf("Trying to open star kdtree %s instead...\n", fn);
			skdt = startree_open(fn);
			if (!skdt) {
				fprintf(stderr, "Failed to read star kdtree %s.\n", fn);
				continue;
			}
			Nstars = startree_N(skdt);
		}

        dimquads = quadfile_dimquads(qf);

		printf("Counting stars in quads...\n");
		starcounts = calloc(sizeof(unsigned char), Nstars);
		for (i=0; i<qf->numquads; i++) {
			unsigned int stars[dimquads];
			int j;
			if (!(i % 200000)) {
				printf(".");
				fflush(stdout);
			}
			quadfile_get_stars(qf, i, stars);
			for (j=0; j<dimquads; j++) {
				assert(stars[j] < Nstars);
				assert(starcounts[stars[j]] < 255);
				starcounts[stars[j]]++;
			}
		}
		printf("\n");


		printf("Computing image...\n");
		for (i=0; i<Nstars; i++) {
			double* xyz;
			double starpos[3];
			double px, py;
			int X, Y;
			int j;
			if (!(i % 100000)) {
				printf(".");
				fflush(stdout);
			}
			if (!starcounts[i])
				continue;
			if (cat)
				xyz = catalog_get_star(cat, i);
			else {
				startree_get(skdt, i, starpos);
				xyz = starpos;
			}
			project_hammer_aitoff_x(xyz[0], xyz[1], xyz[2], &px, &py);
			px = 0.5 + (px - 0.5) * 0.99;
			py = 0.5 + (py - 0.5) * 0.99;
			for (j=0; j<Nimgs; j++) {
				N = il_get(imgsizes, j);
				X = (int)nearbyint(px * N);
				Y = (int)nearbyint(py * N);
				(counts[j])[Y*N + X] += starcounts[i];
			}
		}
		printf("\n");

		if (cat)
			catalog_close(cat);
		if (skdt)
			startree_close(skdt);

		quadfile_close(qf);

		free(starcounts);
	}

	for (i=0; i<Nimgs; i++) {
		FILE* fid;
		int j;
		outfn = pl_get(outnames, i);
		N = il_get(imgsizes, i);

		maxval = 0;
		for (j=0; j<(N*N); j++)
			if (counts[i][j] > maxval)
				maxval = counts[i][j];
		printf("maxval is %i.\n", maxval);

		fid = fopen(outfn, "wb");
		if (!fid) {
			fprintf(stderr, "Couldn't open file %s to write image: %s\n", outfn, strerror(errno));
			exit(-1);
		}
		img = realloc(img, N*N);
		if (!img) {
			fprintf(stderr, "Couldn't allocate %ix%i image.\n", N, N);
			exit(-1);
		}
		for (j=0; j<(N*N); j++)
			img[j] = (int)rint((255.0 * (double)counts[i][j]) / (double)maxval);

		fprintf(fid, "P5 %d %d %d\n",N,N, 255);
		if (fwrite(img, 1, N*N, fid) != (N*N)) {
			fprintf(stderr, "Failed to write image file: %s\n", strerror(errno));
			exit(-1);
		}
		fclose(fid);
		free(outfn);
	}

	free(img);
	free(counts);

	return 0;
}
