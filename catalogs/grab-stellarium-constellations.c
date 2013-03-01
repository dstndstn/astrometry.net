/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <string.h>
#include <ctype.h>

#include "starutil.h"
#include "bl.h"
#include "mathutil.h"
#include "an-endian.h"

const char* OPTIONS = "h";

// size of entries in Stellarium's hipparcos.fab file.
static int HIP_SIZE = 15;
// byte offset to the first element in Stellarium's hipparcos.fab file.
static int HIP_OFFSET = 4;

void print_help(char* progname) {
    //boilerplate_help_header(stdout);
    printf("\nUsage: %s\n"
           "\n", progname);
}

static char* const_dirs[] = {
    ".",
    "/usr/share/stellarium/data/sky_cultures/western", // Debian
    "/home/gmaps/usnob-map/execs" // FIXME
};

static char* hipparcos_fn = "hipparcos.fab";
static char* constfn = "constellationship.fab";
static char* hip_dirs[] = {
    ".",
    "/usr/share/stellarium/data", // Debian
    "/home/gmaps/usnob-map/execs"
};

typedef union {
    uint32_t i;
    float    f;
} intfloat;

static void hip_get_radec(unsigned char* hip, int star1,
                          double* ra, double* dec) {
    intfloat ifval;
    ifval.i = *((uint32_t*)(hip + HIP_SIZE * star1));
    v32_letoh(&ifval.i);
    *ra = ifval.f;
    // Stellarium stores RA in hours...
    *ra *= (360.0 / 24.0);
    ifval.i = *((uint32_t*)(hip + HIP_SIZE * star1 + 4));
    v32_letoh(&ifval.i);
    *dec = ifval.f;
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int c;
    FILE* fconst = NULL;
    uint32_t nstars;
    size_t mapsize;
    void* map;
    unsigned char* hip;
    FILE* fhip = NULL;
    int i;
	pl* cstars;
	il* alluniqstars;
	sl* shortnames;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        }
    }

    if (optind != argc) {
        print_help(args[0]);
        exit(-1);
    }

	for (i=0; i<sizeof(const_dirs)/sizeof(char*); i++) {
		char fn[256];
		snprintf(fn, sizeof(fn), "%s/%s", const_dirs[i], constfn);
		fprintf(stderr, "render_constellation: Trying file: %s\n", fn);
		fconst = fopen(fn, "rb");
		if (fconst)
			break;
	}
	if (!fconst) {
		fprintf(stderr, "render_constellation: couldn't open any constellation files.\n");
		return -1;
	}

	for (i=0; i<sizeof(hip_dirs)/sizeof(char*); i++) {
		char fn[256];
		snprintf(fn, sizeof(fn), "%s/%s", hip_dirs[i], hipparcos_fn);
		fprintf(stderr, "render_constellation: Trying hip file: %s\n", fn);
		fhip = fopen(fn, "rb");
		if (fhip)
			break;
	}
	if (!fhip) {
		fprintf(stderr, "render_constellation: unhip\n");
		return -1;
	}

	// first 32-bit int: 
	if (fread(&nstars, 4, 1, fhip) != 1) {
		fprintf(stderr, "render_constellation: failed to read nstars.\n");
		return -1;
	}
	v32_letoh(&nstars);
	fprintf(stderr, "render_constellation: Found %i Hipparcos stars\n", nstars);

	mapsize = nstars * HIP_SIZE + HIP_OFFSET;
	map = mmap(0, mapsize, PROT_READ, MAP_SHARED, fileno(fhip), 0);
	hip = ((unsigned char*)map) + HIP_OFFSET;

	// for each constellation, its il* of lines.
	cstars = pl_new(16);
	alluniqstars = il_new(16);
	shortnames = sl_new(16);

	for (c=0;; c++) {
		char shortname[16];
		int nlines;
		int i;
		il* stars;

		if (feof(fconst))
			break;

		if (fscanf(fconst, "%s %d ", shortname, &nlines) != 2) {
			fprintf(stderr, "failed to parse name+nlines (constellation %i)\n", c);
			fprintf(stderr, "file offset: %i (%x)\n",
					(int)ftello(fconst), (int)ftello(fconst));
			return -1;
		}
		//fprintf(stderr, "Name: %s.  Nlines %i.\n", shortname, nlines);

		stars = il_new(16);

		sl_append(shortnames, shortname);
		pl_append(cstars, stars);

		for (i=0; i<nlines; i++) {
			int star1, star2;

			if (fscanf(fconst, " %d %d", &star1, &star2) != 2) {
				fprintf(stderr, "failed parse star1+star2\n");
				return -1;
			}

			il_insert_unique_ascending(alluniqstars, star1);
			il_insert_unique_ascending(alluniqstars, star2);

			il_append(stars, star1);
			il_append(stars, star2);
		}
		fscanf(fconst, "\n");
	}
	fprintf(stderr, "render_constellations: Read %i constellations.\n", c);

	printf("static const int constellations_N = %i;\n", sl_size(shortnames));

	/*
	  for (c=0; c<sl_size(shortnames); c++) {
	  printf("static const char* shortname_%i = \"%s\";\n", c, sl_get(shortnames, c));
	  }
	  printf("static const char* shortnames[] = {");
	  for (c=0; c<sl_size(shortnames); c++) {
	  printf("shortname_%i,", c);
	  }
	  printf("};\n");
	*/
	printf("static const char* shortnames[] = {");
	for (c=0; c<sl_size(shortnames); c++) {
		printf("\"%s\",", sl_get(shortnames, c));
	}
	printf("};\n");

	printf("static const int constellation_nlines[] = {");
	for (c=0; c<pl_size(cstars); c++) {
		il* stars = pl_get(cstars, c);
		printf("%i,", il_size(stars)/2);
	}
	printf("};\n");

	for (c=0; c<pl_size(cstars); c++) {
		il* stars = pl_get(cstars, c);
		printf("static const int constellation_lines_%i[] = {", c);
		for (i=0; i<il_size(stars); i++) {
			int s = il_get(stars, i);
			int ms = il_index_of(alluniqstars, s);
			printf("%s%i", (i?",":""), ms);
		}
		printf("};\n");
	}

	printf("static const int* constellation_lines[] = {");
	for (c=0; c<pl_size(cstars); c++) {
		printf("constellation_lines_%i,", c);
	}
	printf("};\n");

	printf("static const int stars_N = %i;\n", il_size(alluniqstars));

	printf("static const double star_positions[] = {");
	for (i=0; i<il_size(alluniqstars); i++) {
		int s = il_get(alluniqstars, i);
		double ra, dec;
		hip_get_radec(hip, s, &ra, &dec);
		printf("%g,%g,", ra, dec);
	}
	printf("};\n");

	munmap(map, mapsize);
	
	fclose(fconst);
	fclose(fhip);

	return 0;
}
