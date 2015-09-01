/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "healpix.h"
#include "starutil.h"
#include "rdlist.h"

int convert_file(char* infn, char* outfn)
{
	int i, j, numfields, npoints, healpixes[12];
	FILE* hpf;
	rdlist* rdls;

	fprintf(stderr, "Reading input from RDLS file %s, writing output to HPLS file %s.\n", infn, outfn);

	// Open the two files for input and output
	rdls = rdlist_open(infn);
	if (!rdls) {
		fprintf(stderr, "Couldn't open RDLS %s.\n", infn);
		return 1;
	}

	hpf = fopen(outfn, "w");
	if (!hpf) {
		fprintf(stderr, "Couldn't open %s for writing: %s\n", outfn, strerror(errno));
		return 1;
	}

	// First line: numfields
	numfields = rdlist_n_fields(rdls);
	fprintf(hpf, "NumFields=%i\n", numfields);

	for (j=1; j<=numfields; j++) {
		int first = 1;
		// Second line and subsequent lines: npoints,ra,dec,ra,dec,...
		dl* points = rdlist_get_field(rdls, j);
		if (!points) {
			fprintf(stderr, "Failed to read RDLS field %i.\n", j);
			return 1;
		}

		for (i = 0; i < 12; i++) {
			healpixes[i] = 0;
		}

		npoints = dl_size(points) / 2;

		for (i = 0; i < npoints; i++) {
			double ra, dec;
			int hp;

			ra  = dl_get(points, i*2);
			dec = dl_get(points, i*2 + 1);

			ra=deg2rad(ra);
			dec=deg2rad(dec);

			hp = radectohealpix(ra, dec, 1);
			if ((hp < 0) || (hp >= 12)) {
				printf("ERROR: hp=%i\n", hp);
				exit(-1);
			}
			healpixes[hp] = 1;
		}
		for (i = 0; i < 12; i++) {
			if (healpixes[i]) {
				if (!first)
					fprintf(hpf, " ");
				fprintf(hpf, "%i", i);
				first = 0;
			}
		}
		fprintf(hpf, "\n");
		fflush(hpf);

		dl_free(points);
	}

	rdlist_close(rdls);
	fclose(hpf);
	return 0;
}

int main(int argc, char** args)
{
	int i;
	if (argc == 1 || !(argc % 2)) {
		fprintf(stderr, "Usage: %s <input-rdls-file> <output-hpls-file> [...]\n", args[0]);
		return 1;
	}

	for (i=1; i+1<argc; i+=2)
		convert_file(args[i], args[i+1]);

	return 0;
}
