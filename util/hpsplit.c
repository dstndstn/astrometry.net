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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/param.h>

#include "healpix.h"
#include "healpix-utils.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "boilerplate.h"
#include "fitsioutils.h"
#include "bl.h"
#include "fitstable.h"
#include "ioutils.h"
#include "mathutil.h"

/**
 Accepts a list of input FITS tables, all with exactly the same
 structure, and including RA,Dec columns.

 Accepts a big-healpix Nside, and a margin specified as nmargin
 healpixes of a given small-healpix Nside.
 (OR, healpix_distance_to_radec < margin_deg?)

 Writes an output file for each of the big-healpixes, containing those
 rows that are within (or within range) of the healpix.
 */

const char* OPTIONS = "hvn:r:d:m:o:";

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] <input-FITS-catalog> [...]\n"
		   "    -o <output-filename-pattern>  with %%i printf-pattern\n"
		   "    [-r <ra-column-name>]: name of RA in FITS table (default RA)\n"
		   "    [-d <dec-column-name>]: name of DEC in FITS table (default DEC)\n"
		   "    [-n <healpix Nside>]: default is 1\n"
		   "    [-m <margin in deg>]: add a margin of this many degrees around the healpixes; default 0\n"
		   "    [-v]: +verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

/*
 struct rd_s {
 double ra;
 double dec;
 };
 typedef rd_s radec;
 */

struct cap_s {
	double xyz[3];
	double r2;
};
typedef struct cap_s cap_t;

int main(int argc, char *argv[]) {
    int argchar;
	char* progname = argv[0];
	sl* infns = sl_new(16);
	char* outfnpat = NULL;
	char* racol = "RA";
	char* deccol = "DEC";
	int loglvl = LOG_MSG;
	int nside = 1;
	double margin = 0.0;
	int NHP;
	
	fitstable_t* intable;
	fitstable_t** outtables;

	char** myargs;
	int nmyargs;
	int i;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'o':
			outfnpat = optarg;
			break;
		case 'r':
			racol = optarg;
			break;
		case 'd':
			deccol = optarg;
			break;
		case 'n':
			nside = atoi(optarg);
			break;
		case 'm':
			margin = atof(optarg);
			break;
		case 'v':
			loglvl++;
			break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	nmyargs = argc - optind;
	myargs = argv + optind;

	for (i=0; i<nmyargs; i++)
		sl_append(infns, myargs[i]);
	
	if (!sl_size(infns)) {
		printHelp(progname);
		printf("Need input filenames!\n");
		exit(-1);
	}
	log_init(loglvl);
	fits_use_error_system();

	NHP = 12 * nside * nside;
	logmsg("%i output healpixes\n", NHP);
	outtables = calloc(NHP, sizeof(fitstable_t*));
	assert(outtables);

	cap_t* mincaps = malloc(NHP * sizeof(cap_t));
	cap_t* maxcaps = malloc(NHP * sizeof(cap_t));
	for (i=0; i<NHP; i++) {
		// center
		double r2;
		double xyz[3];
		double* cxyz;
		double step = 1e-3;
		double v;
		double r2b, r2a;

		cxyz = mincaps[i].xyz;
		healpix_to_xyzarr(i, nside, 0.5, 0.5, mincaps[i].xyz);
		memcpy(maxcaps[i].xyz, cxyz, 3 * sizeof(double));
		// radius-squared:
		// max is the easy one: max of the four corners (I assume)
		r2 = 0.0;
		healpix_to_xyzarr(i, nside, 0.0, 0.0, xyz);
		r2 = MAX(r2, distsq(xyz, cxyz, 3));
		healpix_to_xyzarr(i, nside, 1.0, 0.0, xyz);
		r2 = MAX(r2, distsq(xyz, cxyz, 3));
		healpix_to_xyzarr(i, nside, 0.0, 1.0, xyz);
		r2 = MAX(r2, distsq(xyz, cxyz, 3));
		healpix_to_xyzarr(i, nside, 1.0, 1.0, xyz);
		r2 = MAX(r2, distsq(xyz, cxyz, 3));
		maxcaps[i].r2 = r2;
		r2a = r2;

		r2 = 1.0;
		r2b = 0.0;
		for (v=0; v<=1.0; v+=step) {
			healpix_to_xyzarr(i, nside, 0.0, v, xyz);
			r2 = MIN(r2, distsq(xyz, cxyz, 3));
			r2b = MAX(r2b, distsq(xyz, cxyz, 3));
			healpix_to_xyzarr(i, nside, 1.0, v, xyz);
			r2 = MIN(r2, distsq(xyz, cxyz, 3));
			r2b = MAX(r2b, distsq(xyz, cxyz, 3));
			healpix_to_xyzarr(i, nside, v, 0.0, xyz);
			r2 = MIN(r2, distsq(xyz, cxyz, 3));
			r2b = MAX(r2b, distsq(xyz, cxyz, 3));
			healpix_to_xyzarr(i, nside, v, 1.0, xyz);
			r2 = MIN(r2, distsq(xyz, cxyz, 3));
			r2b = MAX(r2b, distsq(xyz, cxyz, 3));
		}
		logverb("\nhealpix %i: min rad    %g\n", i, sqrt(r2));
		logverb("healpix %i: max rad    %g\n", i, sqrt(r2a));
		logverb("healpix %i: max rad(b) %g\n", i, sqrt(r2b));
		assert(r2a >= r2b);
	}





	for (i=0; i<sl_size(infns); i++) {
		char* infn = sl_get(infns, i);
		int r, NR;
		double radec[2];
		tfits_type any, dubl;
		il* hps = NULL;

		logmsg("Reading input \"%s\"...\n", infn);

		intable = fitstable_open(infn);
		if (!intable) {
			ERROR("Couldn't read catalog %s", infn);
			exit(-1);
		}
		NR = fitstable_nrows(intable);
		logmsg("Got %i rows\n", NR);

		any = fitscolumn_any_type();
		dubl = fitscolumn_double_type();

		fitstable_add_read_column_struct(intable, dubl, 1, 0, any, racol, TRUE);
		fitstable_add_read_column_struct(intable, dubl, 1, sizeof(double), any, deccol, TRUE);

		if (fitstable_read_extension(intable, 1)) {
			ERROR("Failed to find RA and DEC columns (called \"%s\" and \"%s\" in the FITS file)", racol, deccol);
			exit(-1);
		}

		for (r=0; r<NR; r++) {
			int hp = -1;
			double ra, dec;
			int j;
			if (fitstable_read_struct(intable, r, radec)) {
				ERROR("Failed to read row %i from file \"%s\"", r, infn);
				exit(-1);
			}
			ra = radec[0];
			dec = radec[1];
			logverb("row %i: ra,dec %g,%g\n", r, ra, dec);
			if (margin == 0) {
				hp = radecdegtohealpix(ra, dec, nside);
				logverb("  --> healpix %i\n", hp);
			} else {
				hps = healpix_rangesearch_radec(ra, dec, margin, nside, hps);
				logverb("  --> healpixes: [");
				for (j=0; j<il_size(hps); j++)
					logverb(" %i", il_get(hps, j));
				logverb(" ]\n");
			}

			j=0;
			while (1) {
				if (hps) {
					if (j >= il_size(hps))
						break;
					hp = il_get(hps, j);
					j++;
				}
				assert(hp < NHP);
				assert(hp >= 0);

				if (!outtables[hp]) {
					char* outfn;
					fitstable_t* out;
					// MEMLEAK
					asprintf_safe(&outfn, outfnpat, hp);
					logmsg("Opening output file \"%s\"...\n", outfn);
					out = fitstable_open_for_writing(outfn);
					if (!out) {
						ERROR("Failed to open output table \"%s\"", outfn);
						exit(-1);
					}
					// Set the output table structure.
					fitstable_add_fits_columns_as_struct2(intable, out);
					if (fitstable_write_primary_header(out) ||
						fitstable_write_header(out)) {
						ERROR("Failed to write output file headers for \"%s\"", outfn);
						exit(-1);
					}
					outtables[hp] = out;
				}

				if (fitstable_copy_row_data(intable, r, outtables[hp])) {
					ERROR("Failed to copy a row of data from input table \"%s\" to output healpix %i", infn, hp);
					exit(-1);
				}
				if (!hps)
					break;
			}
			if (hps)
				il_remove_all(hps);
			//if (fitstable_read_row_data(intable, r, 
			//fitstable_copy_row_data()
		}
		fitstable_close(intable);
	}

	for (i=0; i<NHP; i++) {
		if (!outtables[i])
			continue;
		if (fitstable_fix_header(outtables[i]) ||
			fitstable_fix_primary_header(outtables[i]) ||
			fitstable_close(outtables[i])) {
			ERROR("Failed to close output table for healpix %i", i);
			exit(-1);
		}
	}

	free(outtables);
	sl_free2(infns);
    return 0;
}



