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
#include <assert.h>
#include <math.h>

#include "tycho2-fits.h"
#include "starutil.h"
#include "mathutil.h"
#include "an-endian.h"

#define OPTIONS "ho:"

static void print_help(char* progname) {
    printf("usage:\n"
		   "  %s -o <output-filename>\n"
		   "  <input-file> [<input-file> ...]\n\n"
		   "Input files must be Tycho-2 FITS files.\n"
		   "Output file will be in Stellarium format.\n\n",
		   progname);
}

static int write_32(FILE* fout, void* p) {
	if (fwrite(p, 1, 4, fout) != 4) {
		fprintf(stderr, "Failed to write 32-bit quantity: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}
static int write_16(FILE* fout, void* p) {
	if (fwrite(p, 1, 2, fout) != 2) {
		fprintf(stderr, "Failed to write 16-bit quantity: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}
static int write_8(FILE* fout, void* p) {
	if (fwrite(p, 1, 1, fout) != 1) {
		fprintf(stderr, "Failed to write 8-bit quantity: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	char* outfn = NULL;
	int c;
	int startoptind;
	FILE* fout;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case '?':
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'o':
			outfn = optarg;
			break;
		}
    }

	if (!outfn || (optind == argc)) {
		print_help(args[0]);
		exit(-1);
	}

	fout = fopen(outfn, "wb");
	if (!fout) {
		fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
		exit(-1);
	}

	//qfits_err_register(qfits_dispfn);
	//qfits_err_statset(1);

	startoptind = optind;
	for (; optind<argc; optind++) {
		tycho2_fits* tycho = NULL;
		qfits_header* hdr;
		anbool is_tycho = FALSE;
		int i, N;
		uint32_t ui;
		char* infn;

		infn = args[optind];
		printf("Opening catalog file %s...\n", infn);
		hdr = qfits_header_read(infn);
		if (!hdr) {
			fprintf(stderr, "Couldn't read FITS header in file %s.\n", infn);
			exit(-1);
		}
		is_tycho = qfits_header_getboolean(hdr, "TYCHO_2", 0);
		if (!is_tycho) {
			fprintf(stderr, "File %s doesn't have TYCHO_2 FITS header.  Skipping file.\n", infn);
			continue;
		}
		tycho = tycho2_fits_open(infn);
		if (!tycho) {
			fprintf(stderr, "Couldn't open Tycho-2 catalog: %s\n", infn);
			exit(-1);
		}

		N = tycho2_fits_count_entries(tycho);

		fprintf(stderr, "Reading %i entries...\n", N);

		// Stellarium header: how many stars in the catalog
		// (32-bit uint).
		ui = u32_htole(N);
		if (write_32(fout, &ui)) exit(-1);

		for (i=0; i<N; i++) {
			tycho2_entry* e;
			float ra, dec, distance, mag;
			int nmag;
			int tmpmag;
			uint16_t imag;
			uint8_t type;

			e = tycho2_fits_read_entry(tycho);
			if (!e) {
				fprintf(stderr, "Failed to read a Tycho-2 entry.\n");
				exit(-1);
			}

			ra = e->RA;
			dec = e->DEC;
			// just avg the available magnitudes.
			nmag = 0;
			mag = 0.0;
			// in Tycho-2, mag 0.0 means it's unavailable.
			if (e->mag_BT != 0.0) {
				mag += e->mag_BT;
				nmag++;
			}
			if (e->mag_VT != 0.0) {
				mag += e->mag_VT;
				nmag++;
			}
			if (e->mag_HP != 0.0) {
				mag += e->mag_HP;
				nmag++;
			}
			if (nmag)
				mag /= (float)nmag;

			// ra: degrees -> hours
			ra *= 24.0 / 360.0;
			// dec: degrees -> degrees
			// imag: 256 * mag - 5

			// distance info isn't in Tycho-2.
			distance = 1.0;

			tmpmag = (int)rint(256.0 * mag);
			if (tmpmag < 0)
				tmpmag += 0x10000;
			imag = tmpmag;

			//type = 12; // "other"
			type = 0; // "other"

            v32_htole(&ra);
            v32_htole(&dec);
            v16_htole(&imag);
            v32_htole(&distance);

			write_32(fout, &ra);
			write_32(fout, &dec);
			write_16(fout, &imag);
			write_8 (fout, &type);
			write_32(fout, &distance);

		}
		tycho2_fits_close(tycho);
	}

	if (fclose(fout)) {
		fprintf(stderr, "Failed to close output file: %s\n", strerror(errno));
		exit(-1);
	}

	return 0;
}
