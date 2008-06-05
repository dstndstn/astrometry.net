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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "fitsioutils.h"
#include "errors.h"
#include "boilerplate.h"
#include "ioutils.h"
#include "starutil.h"

#define OPTIONS "hq:c:s:Q:C:S:"

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
	       "   -q <input-quad-filename>\n"
	       "   -c <input-code-kdtree-filename>\n"
	       "   -s <input-star-kdtree-filename>\n"
	       "   -Q <output-quad-filename>\n"
	       "   -C <output-code-kdtree-filename>\n"
	       "   -S <output-star-kdtree-filename>\n"
	       "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **args) {
    int argchar;
    quadfile* quad;
	codetree* code;
	startree_t* star;
	char* progname = args[0];
	char* quadfn = NULL;
	char* codefn = NULL;
	char* starfn = NULL;

	char* quadoutfn = NULL;
	char* codeoutfn = NULL;
	char* staroutfn = NULL;
    quadfile* quadout;
	int i;
    int N;
    qfits_header* hdr;
    qfits_header* inhdr;

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
        case 'Q':
            quadoutfn = optarg;
            break;
        case 'C':
            codeoutfn = optarg;
            break;
        case 'S':
            staroutfn = optarg;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!(quadfn && starfn && codefn && quadoutfn && staroutfn && codeoutfn)) {
		printHelp(progname);
        fprintf(stderr, "\nYou must specify all filenames (-q, -c, -s, -Q, -C, -S)\n");
		exit(-1);
	}

    fits_use_error_system();

    printf("Ensuring the files are the right types...\n");

	printf("Reading code tree from %s ...\n", codefn);
	code = codetree_open(codefn);
	if (!code) {
		fprintf(stderr, "Failed to read code kdtree from %s.\n", codefn);
		exit(-1);
	}
    printf("Ok.\n");

	printf("Reading star tree from %s ...\n", starfn);
	star = startree_open(starfn);
	if (!star) {
		fprintf(stderr, "Failed to read star kdtree from %s.\n", starfn);
		exit(-1);
	}
    printf("Ok.\n");

	printf("Reading quads from %s ...\n", quadfn);
	quad = quadfile_open(quadfn);
	if (!quad) {
		fprintf(stderr, "Failed to read quads from %s.\n", quadfn);
		exit(-1);
	}
    printf("Ok.\n");


    printf("Writing code tree to %s ...\n", codeoutfn);
    hdr = qfits_header_new();
    inhdr = codetree_header(code);
    fits_append_all_headers(inhdr, hdr, "HISTORY");
    fits_append_all_headers(inhdr, hdr, "COMMENT");
    code->header = hdr;
    if (codetree_write_to_file_flipped(code, codeoutfn)) {
        ERROR("Failed to write code kdtree to file %s", codeoutfn);
        exit(-1);
    }
    codetree_close(code);

    printf("Writing star tree to %s ...\n", staroutfn);
    hdr = qfits_header_new();
    inhdr = startree_header(star);
    fits_append_all_headers(inhdr, hdr, "HISTORY");
    fits_append_all_headers(inhdr, hdr, "COMMENT");
    star->header = hdr;
    if (startree_write_to_file_flipped(star, staroutfn)) {
        ERROR("Failed to write star kdtree to file %s", staroutfn);
        exit(-1);
    }
    startree_close(star);

    printf("Writing quads to %s ...\n", quadoutfn);
    quadout = quadfile_open_for_writing(quadoutfn);
    if (!quadout) {
        ERROR("Failed to open file %s for writing quad file", quadoutfn);
        exit(-1);
    }
    if (quadfile_write_header(quadout)) {
        ERROR("Failed to write quad header");
        exit(-1);
    }
    N = quadfile_nquads(quad);
    for (i=0; i<N; i++) {
        unsigned int q[DQMAX];
        if (quadfile_get_stars(quad, i, q)) {
            ERROR("Failed to read quad %i", i);
            exit(-1);
        }
        if (quadfile_write_quad_flipped(quadout, q)) {
            ERROR("Failed to write quad %i", i);
            exit(-1);
        }
    }
    if (quadfile_fix_header(quadout) ||
        quadfile_close(quadout)) {
        ERROR("Failed to close output quad file");
        exit(-1);
    }
    quadfile_close(quad);

    printf("Done!\n");

	return 0;
}
