/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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
 Merges .quad, .ckdt, and .skdt files to produce a .ind file.
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "fitsioutils.h"
#include "errors.h"
#include "boilerplate.h"
#include "ioutils.h"

#define OPTIONS "hq:c:s:o:"

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
           "   -q <input-quad-filename>\n"
           "   -c <input-code-kdtree-filename>\n"
           "   -s <input-star-kdtree-filename>\n"
           "   -o <output-index-filename>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **args) {
    int argchar;
    quadfile* quad;
	codetree* code;
	startree_t* star;
    FILE* fout;
	FILE* fin;
	char* progname = args[0];
	char* quadfn = NULL;
	char* codefn = NULL;
	char* starfn = NULL;
	char* outfn = NULL;
	int i;
    int N;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'q':
            quadfn = optarg;
            break;
        case 'c':
            codefn = optarg;
            break;
        case 's':
            starfn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!(quadfn && starfn && codefn && outfn)) {
		printHelp(progname);
        fprintf(stderr, "\nYou must specify all filenames (-q, -c, -s, -o)\n");
		exit(-1);
	}

    fits_use_error_system();

    fout = fopen(outfn, "wb");
    if (!fout) {
        SYSERROR("Failed to open output file");
        exit(-1);
    }

    printf("Ensuring the files are the right types...\n");

	printf("Reading code tree from %s ...\n", codefn);
	code = codetree_open(codefn);
	if (!code) {
		fprintf(stderr, "Failed to read code kdtree from %s.\n", codefn);
		exit(-1);
	}
    printf("Ok.\n");
    codetree_close(code);

	printf("Reading star tree from %s ...\n", starfn);
	star = startree_open(starfn);
	if (!star) {
		fprintf(stderr, "Failed to read star kdtree from %s.\n", starfn);
		exit(-1);
	}
    printf("Ok.\n");
    startree_close(star);

	printf("Reading quads from %s ...\n", quadfn);
	quad = quadfile_open(quadfn);
	if (!quad) {
		fprintf(stderr, "Failed to read quads from %s.\n", quadfn);
		exit(-1);
	}
    printf("Ok.\n");
    quadfile_close(quad);

    {
        char* infiles[] = { quadfn, codefn, starfn };
        int j;

        for (j=0; j<3; j++) {
            char* fn = infiles[j];
            printf("Copying %s...\n", fn);

            fin = fopen(fn, "rb");
            if (!fin) {
                SYSERROR("Failed to open input file");
                exit(-1);
            }
            N = qfits_query_n_ext(fn);
            for (i=0; i<=N; i++) {
                int hdrstart, hdrlen, datastart, datalen;
                // skip the primary headers of code and star trees.
                if (j != 0 && i == 0)
                    continue;
                if (qfits_get_hdrinfo(fn, i, &hdrstart,  &hdrlen ) ||
                    qfits_get_datinfo(fn, i, &datastart, &datalen)) {
                    fprintf(stderr, "Error getting extents of extension %i.\n", i);
                    exit(-1);
                }
                if (pipe_file_offset(fin, hdrstart,  hdrlen,  fout) ||
                    pipe_file_offset(fin, datastart, datalen, fout)) {
                    ERROR("Error writing extension %i", i);
                }
            }
            fclose(fin);
        }
    }

    if (fclose(fout)) {
        SYSERROR("Failed to close output file %s", outfn);
        exit(-1);
    }

	return 0;
}
