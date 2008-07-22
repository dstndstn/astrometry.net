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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <assert.h>

#include "nomad.h"
#include "nomad-fits.h"
#include "qfits.h"
#include "healpix.h"
#include "starutil.h"
#include "fitsioutils.h"
#include "boilerplate.h"

#define OPTIONS "ho:N:"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage:\n"
		   "  %s -o <output-filename-template>     [eg, nomad_%%03i.fits]\n"
		   "  [-N <healpix-nside>]  (default = 9)\n"
		   "  <input-file> [<input-file> ...]\n"
		   "\n"
		   "The output-filename-template should contain a \"printf\" sequence like \"%%03i\";\n"
		   "we use \"sprintf(filename, output-filename-template, healpix)\" to determine the filename\n"
		   "to be used for each healpix.\n\n"
		   "\nNOTE: WE ASSUME THE NOMAD FILES ARE GIVEN ON THE COMMAND LINE IN ORDER: 000/b0000.cat, 000/b0001.cat, etc.\n\n\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	char* outfn = NULL;
    int c;
	int startoptind;
	int nrecords, nfiles;
	int Nside = 9;

	nomad_fits** nomads;

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

	nomads = calloc(HP, sizeof(nomad_fits*));

	memset(slicecounts, 0, 1800 * sizeof(uint));

	nrecords = 0;
	nfiles = 0;

	printf("Reading NOMAD files... ");
	fflush(stdout);

	startoptind = optind;
	for (; optind<argc; optind++) {
		char* infn;
		FILE* fid;
		unsigned char* map;
		size_t map_size;
		int i;
		int lastgrass;

		infn = args[optind];
		fid = fopen(infn, "rb");
		if (!fid) {
			fprintf(stderr, "Couldn't open input file %s: %s\n", infn, strerror(errno));
			exit(-1);
		}

		if ((optind > startoptind) && ((optind - startoptind) % 100 == 0)) {
			printf("\nReading file %i of %i: %s\n", optind - startoptind,
				   argc - startoptind, infn);
		}

		if (fseeko(fid, 0, SEEK_END)) {
			fprintf(stderr, "Couldn't seek to end of input file %s: %s\n", infn, strerror(errno));
			exit(-1);
		}

		map_size = ftello(fid);

		fseeko(fid, 0, SEEK_SET);

		map = mmap(NULL, map_size, PROT_READ, MAP_SHARED, fileno(fid), 0);
		if (map == MAP_FAILED) {
			fprintf(stderr, "Couldn't mmap input file %s: %s\n", infn, strerror(errno));
			exit(-1);
		}
		fclose(fid);

		if (map_size % NOMAD_RECORD_SIZE) {
			fprintf(stderr, "Warning, input file %s has size %u which is not divisible into %i-byte records.\n",
					infn, (unsigned int)map_size, NOMAD_RECORD_SIZE);
		}

		printf("File %i of %i: %s: %i records.\n", optind - startoptind, argc - startoptind, infn, (int)(map_size / NOMAD_RECORD_SIZE));

		lastgrass = 0;
		for (i=0; i<map_size; i+=NOMAD_RECORD_SIZE) {
			nomad_entry entry;
			int hp;
			int slice;

			if ((i * 80 / map_size) != lastgrass) {
				printf(".");
				fflush(stdout);
				lastgrass = i * 80 / map_size;
			}
			
			if (nomad_parse_entry(&entry, map + i)) {
				fprintf(stderr, "Failed to parse NOMAD entry: offset %i in file %s.\n",
						i, infn);
				exit(-1);
			}

			// compute the nomad_id based on its DEC slice and index.
			slice = (int)(10.0 * (entry.dec + 90.0));
			assert(slice < 1800);
			assert((slicecounts[slice] & 0xffe00000) == 0);
			entry.nomad_id = (slice << 21) | (slicecounts[slice]);
			slicecounts[slice]++;

			hp = radectohealpix(deg2rad(entry.ra), deg2rad(entry.dec), Nside);

			if (!nomads[hp]) {
				char fn[256];
				sprintf(fn, outfn, hp);
				nomads[hp] = nomad_fits_open_for_writing(fn);
				if (!nomads[hp]) {
					fprintf(stderr, "Failed to initialized FITS file %i (filename %s).\n", hp, fn);
					exit(-1);
				}

				// header remarks...
				fits_header_add_int(nomads[hp]->header, "HEALPIX", hp, "The healpix number of this catalog.");
				fits_header_add_int(nomads[hp]->header, "NSIDE", Nside, "The healpix resolution.");
				boilerplate_add_fits_headers(nomads[hp]->header);
				qfits_header_add(nomads[hp]->header, "HISTORY", "Created by the program \"nomadtofits\"", NULL, NULL);
				qfits_header_add(nomads[hp]->header, "HISTORY", "nomadtofits command line:", NULL, NULL);
				fits_add_args(nomads[hp]->header, args, argc);
				qfits_header_add(nomads[hp]->header, "HISTORY", "(end of command line)", NULL, NULL);

				if (nomad_fits_write_headers(nomads[hp])) {
					fprintf(stderr, "Failed to write header for FITS file %s.\n", fn);
					exit(-1);
				}
			}

			if (nomad_fits_write_entry(nomads[hp], &entry)) {
				fprintf(stderr, "Failed to write FITS entry.\n");
				exit(-1);
			}

			nrecords++;
		}

		munmap(map, map_size);

		nfiles++;
		printf("\n");
	}
	printf("\n");

	// close all the files...
	for (i=0; i<HP; i++) {
		if (!nomads[i])
			continue;
		if (nomad_fits_fix_headers(nomads[i]) ||
			nomad_fits_close(nomads[i])) {
			fprintf(stderr, "Failed to close file %i: %s\n", i, strerror(errno));
		}
	}

	printf("Read %u files, %u records.\n", nfiles, nrecords);

	free(nomads);

	return 0;
}

