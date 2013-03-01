/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.

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
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>

#include <bzlib.h>

#include "ucac3-fits.h"
#include "ucac3.h"
#include "qfits.h"
#include "healpix.h"
#include "starutil.h"
#include "fitsioutils.h"
#include "log.h"
#include "errors.h"
#include "boilerplate.h"

#define OPTIONS "ho:N:"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage:\n"
		   "  %s -o <output-filename-template>     [default: ucac3_%%03i.fits]\n"
		   "  [-N <healpix-nside>]  (default = 9)\n"
		   "  <input-file> [<input-file> ...]\n"
		   "\n"
		   "The output-filename-template should contain a \"printf\" sequence like \"%%03i\";\n"
		   "we use \"sprintf(filename, output-filename-template, healpix)\" to determine the filename\n"
		   "to be used for each healpix.\n\n"
		   "\nNOTE: WE ASSUME THE UCAC3 FILES ARE GIVEN ON THE COMMAND LINE IN ORDER: z001.bz2, z002.bz2, ..., z360.bz2.\n\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

#define CHECK_BZERR() \
	do { if (bzerr != BZ_OK) { ERROR("bzip2 error: code %i", bzerr);	\
			BZ2_bzReadClose(&bzerr, bzfid); exit(-1); }} while (0);

int main(int argc, char** args) {
	char* outfn = "ucac3_%03i.fits";
    int c;
	int startoptind;
	int nrecords, nfiles;
	int Nside = 9;

	ucac3_fits** ucacs;

	int i, HP;
	int slicecounts[1800];

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

	log_init(LOG_MSG);
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
	printf("Nside = %i, using %i healpixes.\n", Nside, HP);
	ucacs = calloc(HP, sizeof(ucac3_fits*));
	memset(slicecounts, 0, 1800 * sizeof(uint));
	nrecords = 0;
	nfiles = 0;

	startoptind = optind;
	for (; optind<argc; optind++) {
		char* infn;
		FILE* fid;
		BZFILE* bzfid;
		int bzerr;
		int i;

		infn = args[optind];
		printf("Reading %s\n", infn);
		if ((optind > startoptind) && ((optind - startoptind) % 100 == 0)) {
			printf("\nReading file %i of %i: %s\n", optind - startoptind,
				   argc - startoptind, infn);
		}
		fflush(stdout);

		fid = fopen(infn, "rb");
		if (!fid) {
			SYSERROR("Couldn't open input file \"%s\"", infn);
			exit(-1);
		}

		// MAGIC 1: bzip verbosity: [0=silent, 4=debug]
		// 0: small -- don't use less memory
		bzfid = BZ2_bzReadOpen(&bzerr, fid, 1, 0, NULL, 0);
		CHECK_BZERR();

		for (i=0;; i++) {
			ucac3_entry entry;
			int hp;
			char buf[UCAC3_RECORD_SIZE];
			int nr;
			anbool eof = 0;

			nr = BZ2_bzRead(&bzerr, bzfid, buf, UCAC3_RECORD_SIZE);
			if ((bzerr == BZ_STREAM_END) && (nr == UCAC3_RECORD_SIZE))
				eof = TRUE;
			else
				CHECK_BZERR();

			if (ucac3_parse_entry(&entry, buf)) {
				ERROR("Failed to parse UCAC3 entry %i in file \"%s\".", i, infn);
				exit(-1);
			}

			hp = radecdegtohealpix(entry.ra, entry.dec, Nside);

			if (!ucacs[hp]) {
				char fn[256];
				sprintf(fn, outfn, hp);
				ucacs[hp] = ucac3_fits_open_for_writing(fn);
				if (!ucacs[hp]) {
					ERROR("Failed to initialize FITS file %i (filename %s)", hp, fn);
					exit(-1);
				}
				fits_header_add_int(ucacs[hp]->header, "HEALPIX", hp, "The healpix number of this catalog.");
				fits_header_add_int(ucacs[hp]->header, "NSIDE", Nside, "The healpix resolution.");
				boilerplate_add_fits_headers(ucacs[hp]->header);
				qfits_header_add(ucacs[hp]->header, "HISTORY", "Created by the program \"ucac3tofits\"", NULL, NULL);
				qfits_header_add(ucacs[hp]->header, "HISTORY", "ucac3tofits command line:", NULL, NULL);
				fits_add_args(ucacs[hp]->header, args, argc);
				qfits_header_add(ucacs[hp]->header, "HISTORY", "(end of command line)", NULL, NULL);
				if (ucac3_fits_write_headers(ucacs[hp])) {
					ERROR("Failed to write header for FITS file %s", fn);
					exit(-1);
				}
			}
			if (ucac3_fits_write_entry(ucacs[hp], &entry)) {
				ERROR("Failed to write FITS entry");
				exit(-1);
			}
			nrecords++;

			if (eof)
				break;
		}

		BZ2_bzReadClose(&bzerr, bzfid);
		fclose(fid);

		nfiles++;
		printf("\n");
	}
	printf("\n");

	// close all the files...
	for (i=0; i<HP; i++) {
		if (!ucacs[i])
			continue;
		if (ucac3_fits_fix_headers(ucacs[i]) ||
			ucac3_fits_close(ucacs[i])) {
			ERROR("Failed to close file %i", i);
		}
	}
	printf("Read %u files, %u records.\n", nfiles, nrecords);
	free(ucacs);
	return 0;
}


