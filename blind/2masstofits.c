/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <zlib.h>

#include "2mass.h"
#include "2mass-fits.h"
#include "healpix.h"
#include "starutil.h"
#include "boilerplate.h"
#include "fitsioutils.h"

#define OPTIONS "ho:N:"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
    printf("usage:\n"
		   "  %s -o <output-filename-template>\n"
		   "  [-N <healpix-nside>]  (default = 8.)\n"
		   "  <input-file> [<input-file> ...]\n"
           "\n"
           "Input files are gzipped 2MASS PSC catalog files, named like psc_aaa.gz."
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int c;
	char* outfn = NULL;
	int startoptind;
	int Nside = 8;
	int HP;
	int i;

	twomass_fits** cats;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case '?':
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'N':
			Nside = atoi(optarg);
			break;
		case 'o':
			outfn = optarg;
			break;
		}
    }

	if (!outfn || (optind == argc)) {
		print_help(args[0]);
		exit(-1);
	}

	if (Nside < 1) {
		fprintf(stderr, "Nside must be >= 1.\n");
		print_help(args[0]);
		exit(-1);
	}

	HP = 12 * Nside * Nside;
	cats = calloc((size_t)HP, sizeof(twomass_fits*));

	printf("Nside = %i, using %i healpixes.\n", Nside, HP);

	printf("Reading 2MASS files... ");
	fflush(stdout);

	startoptind = optind;
	for (; optind<argc; optind++) {
		char* infn;
		gzFile fiz = NULL;
		char line[1024];
		int nentries;

		infn = args[optind];
		printf("\nReading file %i of %i: %s\n", 1 + optind - startoptind,
			   argc - startoptind, infn);
		infn = args[optind];
		fiz = gzopen(infn, "rb");
		if (!fiz) {
			fprintf(stderr, "Failed to open file %s: %s\n", infn, strerror(errno));
			exit(-1);
		}
		nentries = 0;
		for (;;) {
			twomass_entry e;
			int hp;

			if (gzeof(fiz))
				break;

			if (gzgets(fiz, line, 1024) == Z_NULL) {
				if (gzeof(fiz))
					break;
				fprintf(stderr, "Failed to read a line from file %s: %s\n", infn, strerror(errno));
				exit(-1);
			}

			if (twomass_parse_entry(&e, line)) {
				fprintf(stderr, "Failed to parse 2MASS entry from file %s.\n", infn);
				exit(-1);
			}

			hp = radectohealpix(deg2rad(e.ra), deg2rad(e.dec), Nside);
			if (!cats[hp]) {
				char fn[256];
                qfits_header* hdr;

				sprintf(fn, outfn, hp);
				cats[hp] = twomass_fits_open_for_writing(fn);
				if (!cats[hp]) {
					fprintf(stderr, "Failed to open 2MASS catalog for writing to file %s (hp %i).\n", fn, hp);
					exit(-1);
				}
				// header remarks...
                hdr = twomass_fits_get_primary_header(cats[hp]);
				boilerplate_add_fits_headers(hdr);
				fits_header_add_int(hdr, "HEALPIX", hp, "The healpix number of this catalog.");
				fits_header_add_int(hdr, "NSIDE", Nside, "The healpix resolution.");

				fits_add_long_comment(hdr, "The fields are as described in the 2MASS documentation:");
				fits_add_long_comment(hdr, "  ftp://ftp.ipac.caltech.edu/pub/2mass/allsky/format_psc.html");
				fits_add_long_comment(hdr, "with a few exceptions:");
				fits_add_long_comment(hdr, "* all angular fields are measured in degrees");
				fits_add_long_comment(hdr, "* the photometric quality flag values are:");
				fits_add_long_comment(hdr, "    %i: 'X' in 2MASS, No brightness info available.", TWOMASS_QUALITY_NO_BRIGHTNESS);
				fits_add_long_comment(hdr, "    %i: 'U' in 2MASS, The brightness val is an upper bound.", TWOMASS_QUALITY_UPPER_LIMIT_MAG);
				fits_add_long_comment(hdr, "    %i: 'F' in 2MASS, No magnitude sigma is available", TWOMASS_QUALITY_NO_SIGMA);
				fits_add_long_comment(hdr, "    %i: 'E' in 2MASS, Profile-fit photometry was bad", TWOMASS_QUALITY_BAD_FIT);
				fits_add_long_comment(hdr, "    %i: 'A' in 2MASS, Best quality", TWOMASS_QUALITY_A);
				fits_add_long_comment(hdr, "    %i: 'B' in 2MASS, ...", TWOMASS_QUALITY_B);
				fits_add_long_comment(hdr, "    %i: 'C' in 2MASS, ...", TWOMASS_QUALITY_C);
				fits_add_long_comment(hdr, "    %i: 'D' in 2MASS, Worst quality", TWOMASS_QUALITY_D);
				fits_add_long_comment(hdr, "* the confusion/contamination flag values are:");
				fits_add_long_comment(hdr, "    %i: '0' in 2MASS, No problems.", TWOMASS_CC_NONE);
				fits_add_long_comment(hdr, "    %i: 'p' in 2MASS, Persistence.", TWOMASS_CC_PERSISTENCE);
				fits_add_long_comment(hdr, "    %i: 'c' in 2MASS, Confusion.", TWOMASS_CC_CONFUSION);
				fits_add_long_comment(hdr, "    %i: 'd' in 2MASS, Diffraction.", TWOMASS_CC_DIFFRACTION);
				fits_add_long_comment(hdr, "    %i: 's' in 2MASS, Stripe.", TWOMASS_CC_STRIPE);
				fits_add_long_comment(hdr, "    %i: 'b' in 2MASS, Band merge.", TWOMASS_CC_BANDMERGE);
				fits_add_long_comment(hdr, "* the association flag values are:");
				fits_add_long_comment(hdr, "    %i: none.", TWOMASS_ASSOCIATION_NONE);
				fits_add_long_comment(hdr, "    %i: Tycho.", TWOMASS_ASSOCIATION_TYCHO);
				fits_add_long_comment(hdr, "    %i: USNO A-2.", TWOMASS_ASSOCIATION_USNOA2);
				fits_add_long_comment(hdr, "* the NULL value for floats is %f", TWOMASS_NULL);
				fits_add_long_comment(hdr, "* the NULL value for the 'ext_key' aka 'xsc_key' field is");
				fits_add_long_comment(hdr, "   %i (0x%x).", TWOMASS_KEY_NULL, TWOMASS_KEY_NULL);

				if (twomass_fits_write_headers(cats[hp])) {
					fprintf(stderr, "Failed to write 2MASS catalog headers: %s\n", fn);
					exit(-1);
				}
			}
			if (twomass_fits_write_entry(cats[hp], &e)) {
				fprintf(stderr, "Failed to write 2MASS catalog entry.\n");
				exit(-1);
			}

			nentries++;
			if (!(nentries % 100000)) {
				printf(".");
				fflush(stdout);
			}
		}
		gzclose(fiz);
		printf("\n");

		printf("Read %i entries.\n", nentries);
	}

	printf("Finishing up...\n");
	for (i=0; i<HP; i++) {
		if (!cats[i])
			continue;
		if (twomass_fits_fix_headers(cats[i]) ||
			twomass_fits_close(cats[i])) {
			fprintf(stderr, "Failed to close 2MASS catalog.\n");
			exit(-1);
		}
	}
	free(cats);
	printf("Done!\n");

	return 0;
}

