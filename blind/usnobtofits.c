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
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "usnob.h"
#include "qfits.h"
#include "healpix.h"
#include "starutil.h"
#include "usnob-fits.h"
#include "fitsioutils.h"
#include "boilerplate.h"

#define OPTIONS "ho:N:"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage:\n"
		   "  %s -o <output-filename-template>     [eg, usnob10_%%03i.fits]\n"
		   "  [-N <healpix-nside>]  (default = 8)\n"
		   "  <input-file> [<input-file> ...]\n"
		   "\n"
		   "The output-filename-template should contain a \"printf\" sequence like \"%%03i\";\n"
		   "we use \"sprintf(filename, output-filename-template, healpix)\" to determine the filename\n"
		   "to be used for each healpix.\n\n"
		   "\nNOTE: WE ASSUME THE USNO-B1.0 FILES ARE GIVEN ON THE COMMAND LINE IN ORDER: 000/b0000.cat, 000/b0001.cat, etc.\n\n\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	char* outfn = NULL;
    int c;
	int startoptind;
	int nrecords, nobs, nfiles;
	int Nside = 8;

	usnob_fits** usnobs;

	int i, HP;
	int slicecounts[180];

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

	{
		struct rlimit lim;
		getrlimit(RLIMIT_NOFILE, &lim);
		printf("Maximum number of files that can be opened: %li soft, %li hard\n",
			   (long int)lim.rlim_cur, (long int)lim.rlim_max);
		if (lim.rlim_cur < HP) {
			printf("\n\nWARNING: This process is likely to fail - probably after working for many hours!\n\n\n");
			sleep(5);
		}
	}

	usnobs = calloc(HP, sizeof(usnob_fits*));

	memset(slicecounts, 0, 180 * sizeof(uint));

	nrecords = 0;
	nobs = 0;
	nfiles = 0;

	printf("Reading USNO files... ");
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

        printf("Reading file %i of %i: %s\n", optind - startoptind,
               argc - startoptind, infn);

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

		if (map_size % USNOB_RECORD_SIZE) {
			fprintf(stderr, "Warning, input file %s has size %u which is not divisible into %i-byte records.\n",
					infn, (unsigned int)map_size, USNOB_RECORD_SIZE);
		}

        lastgrass = 0;
		for (i=0; i<map_size; i+=USNOB_RECORD_SIZE) {
			usnob_entry entry;
			int hp;
			int slice;

			if ((i * 80 / map_size) != lastgrass) {
				printf(".");
				fflush(stdout);
				lastgrass = i * 80 / map_size;
			}

			if (usnob_parse_entry(map + i, &entry)) {
				fprintf(stderr, "Failed to parse USNOB entry: offset %i in file %s.\n",
						i, infn);
				exit(-1);
			}

			// compute the usnob_id based on its DEC slice and index.
			slice = (int)(entry.dec + 90.0);
			assert(slice < 180);
			assert((slicecounts[slice] & 0xff000000) == 0);
			entry.usnob_id = (slice << 24) | (slicecounts[slice]);
			slicecounts[slice]++;

			hp = radectohealpix(deg2rad(entry.ra), deg2rad(entry.dec), Nside);

			if (!usnobs[hp]) {
				char fn[256];
                qfits_header* hdr;
				sprintf(fn, outfn, hp);
				usnobs[hp] = usnob_fits_open_for_writing(fn);
				if (!usnobs[hp]) {
					fprintf(stderr, "Failed to initialized FITS file %i (filename %s).\n", hp, fn);
					exit(-1);
				}
                if (usnob_fits_remove_an_diffraction_spike_column(usnobs[hp])) {
                    fprintf(stderr, "Failed to remove the AN_DIFFRACTION_SPIKE column.\n");
                    exit(-1);
                }
                hdr = usnob_fits_get_header(usnobs[hp]);
                assert(hdr);

				// header remarks...
				fits_header_add_int(hdr, "HEALPIX", hp, "The healpix number of this catalog.");
				fits_header_add_int(hdr, "NSIDE", Nside, "The healpix resolution.");
				boilerplate_add_fits_headers(hdr);
				qfits_header_add(hdr, "HISTORY", "Created by the program \"usnobtofits\"", NULL, NULL);
				qfits_header_add(hdr, "HISTORY", "usnobtofits command line:", NULL, NULL);
				fits_add_args(hdr, args, argc);
				qfits_header_add(hdr, "HISTORY", "(end of command line)", NULL, NULL);

				if (usnob_fits_write_headers(usnobs[hp])) {
					fprintf(stderr, "Failed to write header for FITS file %s.\n", fn);
					exit(-1);
				}
			}

			if (usnob_fits_write_entry(usnobs[hp], &entry)) {
				fprintf(stderr, "Failed to write FITS entry.\n");
				exit(-1);
			}

			nrecords++;
			nobs += (entry.ndetections == 0 ? 1 : entry.ndetections);
		}

		munmap(map, map_size);

		nfiles++;
		printf("\n");
	}

	// close all the files...
	for (i=0; i<HP; i++) {
		if (!usnobs[i])
			continue;
		if (usnob_fits_fix_headers(usnobs[i]) ||
			usnob_fits_close(usnobs[i])) {
			fprintf(stderr, "Failed to close file %i: %s\n", i, strerror(errno));
		}
	}

	printf("Read %u files, %u records, %u observations.\n",
		   nfiles, nrecords, nobs);

	free(usnobs);

	return 0;
}
