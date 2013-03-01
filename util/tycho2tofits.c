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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <assert.h>

#include "tycho2.h"
#include "tycho2-fits.h"
#include "starutil.h"
#include "healpix.h"
#include "boilerplate.h"
#include "fitsioutils.h"

#define OPTIONS "ho:HN:"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
    printf("\nUsage:\n"
		   "  %s -o <output-filename(-template)>   (eg, tycho2_hp%%02i.fits if you use the -H option)\n"
		   "  [-H]: do healpixification\n"
		   "  [-N <healpix-nside>]\n"
		   "  <input-file> [<input-file> ...]\n\n"
		   "(Healpixification isn't usually necessary because the Tycho-2 catalog is small.)\n\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	char* outfn = NULL;
    int c;
	int nrecords, nobs;
	int Nside = 8;
	tycho2_fits** tycs;
	int i, HP;
	int do_hp = 0;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case '?':
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'H':
			do_hp = 1;
			break;
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

	if (do_hp) {
		HP = 12 * Nside * Nside;
	} else {
		HP = 1;
	}
		
	tycs = malloc(HP * sizeof(tycho2_fits*));
	memset(tycs, 0, HP * sizeof(tycho2_fits*));

	nrecords = 0;
	nobs = 0;

	printf("Reading Tycho-2 files... \n");
	fflush(stdout);

	for (; optind<argc; optind++) {
		char* infn;
		FILE* fid;
		char* map;
		size_t map_size;
		int i;
		anbool supplement;
		int recsize;

		infn = args[optind];
		fid = fopen(infn, "rb");
		if (!fid) {
			fprintf(stderr, "Couldn't open input file %s: %s\n", infn, strerror(errno));
			exit(-1);
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

		supplement = tycho2_guess_is_supplement(map);
		printf("File %s: supplement format: %s\n", infn, (supplement ? "Yes" : "No"));

		if (supplement) {
			recsize = TYCHO_SUPPLEMENT_RECORD_SIZE_RAW;
		} else {
			recsize = TYCHO_RECORD_SIZE_RAW;
		}

		if ((map_size % recsize) && (map_size % (recsize+1)) && (map_size % (recsize+2))) {
			fprintf(stderr, "Warning, input file %s has size %u which is not divisible into %i-, %i-, or %i-byte records.\n",
				infn, (uint)map_size, recsize, recsize+1, recsize+2);
		}

		for (i=0; i<map_size;) {
			tycho2_entry entry;
			int hp;

			if (supplement) {
				if (tycho2_supplement_parse_entry(map + i, &entry)) {
					fprintf(stderr, "Failed to parse TYCHO-2 supplement entry: offset %i in file %s.\n",
							i, infn);
					exit(-1);
				}
			} else {
				if (tycho2_parse_entry(map + i, &entry)) {
					fprintf(stderr, "Failed to parse TYCHO-2 entry: offset %i in file %s.\n",
							i, infn);
					exit(-1);
				}
			}
			//printf("RA, DEC (%g, %g)\n", entry.RA, entry.DEC);

			i += recsize;
			// skip past "\r" and "\n".
			while ((i < map_size) &&
			       ((map[i] == '\r') || (map[i] == '\n')))
				i++;

			if (do_hp) {
				hp = radectohealpix(deg2rad(entry.ra), deg2rad(entry.dec), Nside);
			} else {
				hp = 0;
			}

			if (!tycs[hp]) {
				char fn[256];
                qfits_header* hdr;
				sprintf(fn, outfn, hp);
				tycs[hp] = tycho2_fits_open_for_writing(fn);
				if (!tycs[hp]) {
					fprintf(stderr, "Failed to initialized FITS output file %s.\n", fn);
					exit(-1);
				}
                hdr = tycho2_fits_get_header(tycs[hp]);

				// header remarks...
				qfits_header_add(hdr, "HEALPIXD", (do_hp ? "T" : "F"), "Is this catalog healpixified?", NULL);
				if (do_hp) {
					fits_header_add_int(hdr, "HEALPIX", hp, "The healpix number of this catalog.");
					fits_header_add_int(hdr, "NSIDE", Nside, "The healpix resolution.");
				}

				boilerplate_add_fits_headers(hdr);

				qfits_header_add(hdr, "HISTORY", "Created by the program \"tycho2tofits\"", NULL, NULL);
				qfits_header_add(hdr, "HISTORY", "tycho2tofits command line:", NULL, NULL);
				fits_add_args(hdr, args, argc);
				qfits_header_add(hdr, "HISTORY", "(end of command line)", NULL, NULL);

				if (tycho2_fits_write_headers(tycs[hp])) {
					fprintf(stderr, "Failed to write header for FITS file %s.\n", fn);
					exit(-1);
				}
			}

			if (tycho2_fits_write_entry(tycs[hp], &entry)) {
				fprintf(stderr, "Failed to write Tycho-2 FITS entry.\n");
				exit(-1);
			}

			if (i && ((i/recsize) % 100000 == 0)) {
				printf(".");
				fflush(stdout);
			}

			nrecords++;
			nobs += entry.nobs;
		}

		munmap(map, map_size);

		printf(".");
		fflush(stdout);
	}
	printf("\n");

	// close all the files...
	for (i=0; i<HP; i++) {
		if (!tycs[i])
			continue;
		if (tycho2_fits_fix_headers(tycs[i]) ||
			tycho2_fits_close(tycs[i])) {
			fprintf(stderr, "Failed to close Tycho-2 FITS file.\n");
			exit(-1);
		}
	}
	
	printf("Read %u records, %u observations.\n", nrecords, nobs);
	
	free(tycs);
	return 0;
}

