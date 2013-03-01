/*
  This file is part of the Astrometry.net suite.
  Copyright 2011, 2012, 2013 Dustin Lang.

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
#include <arpa/inet.h>
#include <assert.h>

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

const char* OPTIONS = "hvn:r:d:m:o:gc:t:b:";

void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] <input-FITS-catalog> [...]\n"
		   "    -o <output-filename-pattern>  with %%i printf-pattern\n"
		   "    [-r <ra-column-name>]: name of RA in FITS table (default RA)\n"
		   "    [-d <dec-column-name>]: name of DEC in FITS table (default DEC)\n"
		   "    [-n <healpix Nside>]: default is 1\n"
		   "    [-m <margin in deg>]: add a margin of this many degrees around the healpixes; default 0\n"
		   "    [-g]: gzip'd inputs\n"
		   "    [-c <name>]: copy given column name to the output files\n"
		   "    [-t <temp-dir>]: use the given temp dir; default is /tmp\n"
		   "    [-b <backref-file>]: save the filenumber->filename map in this file; enables writing backreferences too\n"
		   "    [-v]: +verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

struct cap_s {
	double xyz[3];
	double r2;
};
typedef struct cap_s cap_t;

static int refill_rowbuffer(void* baton, void* buffer,
							unsigned int offset, unsigned int nelems) {
	fitstable_t* table = baton;
	//printf("refill_rowbuffer: offset %i, n %i\n", offset, nelems);
	return fitstable_read_nrows_data(table, offset, nelems, buffer);
}


int main(int argc, char *argv[]) {
    int argchar;
	char* progname = argv[0];
	sl* infns = sl_new(16);
	char* outfnpat = NULL;
	char* racol = "RA";
	char* deccol = "DEC";
	char* tempdir = "/tmp";
	anbool gzip = FALSE;
	sl* cols = sl_new(16);
	int loglvl = LOG_MSG;
	int nside = 1;
	double margin = 0.0;
	int NHP;
	double md;
	char* backref = NULL;
	
	fitstable_t* intable;
	fitstable_t** outtables;

	char** myargs;
	int nmyargs;
	int i;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
		case 'b':
			backref = optarg;
			break;
		case 't':
			tempdir = optarg;
			break;
		case 'c':
			sl_append(cols, optarg);
			break;
		case 'g':
			gzip = TRUE;
			break;
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

	if (sl_size(cols) == 0) {
		sl_free2(cols);
		cols = NULL;
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

	md = deg2dist(margin);

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
		maxcaps[i].r2 = square(MAX(0, sqrt(r2) - md));
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
		mincaps[i].r2 = square(MAX(0, sqrt(r2) - md));
		logverb("\nhealpix %i: min rad    %g\n", i, sqrt(r2));
		logverb("healpix %i: max rad    %g\n", i, sqrt(r2a));
		logverb("healpix %i: max rad(b) %g\n", i, sqrt(r2b));
		assert(r2a >= r2b);
	}

	if (backref) {
		fitstable_t* tab = fitstable_open_for_writing(backref);
		int maxlen = 0;
		char* buf;
		for (i=0; i<sl_size(infns); i++) {
			char* infn = sl_get(infns, i);
			maxlen = MAX(maxlen, strlen(infn));
		}
		fitstable_add_write_column_array(tab, fitscolumn_char_type(), maxlen,
										 "filename", NULL);
		fitstable_add_write_column(tab, fitscolumn_i16_type(), "index", NULL);
		if (fitstable_write_primary_header(tab) ||
			fitstable_write_header(tab)) {
			ERROR("Failed to write header of backref table \"%s\"", backref);
			exit(-1);
		}
		buf = malloc(maxlen+1);
		assert(buf);

		for (i=0; i<sl_size(infns); i++) {
			char* infn = sl_get(infns, i);
			int16_t ind;
			memset(buf, 0, maxlen);
			strcpy(buf, infn);
			ind = i;
			if (fitstable_write_row(tab, buf, &ind)) {
				ERROR("Failed to write row %i of backref table: %s = %i",
					  i, buf, ind);
				exit(-1);
			}
		}
		if (fitstable_fix_header(tab) ||
			fitstable_close(tab)) {
			ERROR("Failed to fix header & close backref table");
			exit(-1);
		}
		logmsg("Wrote backref table %s\n", backref);
		free(buf);
	}

	for (i=0; i<sl_size(infns); i++) {
		char* infn = sl_get(infns, i);
		char* originfn = infn;
		int r, NR;
		tfits_type any, dubl;
		il* hps = NULL;
		bread_t* rowbuf;
		int R;
		char* tempfn = NULL;
		char* padrowdata = NULL;
		int ii;

		logmsg("Reading input \"%s\"...\n", infn);

		if (gzip) {
			char* cmd;
			int rtn;
			tempfn = create_temp_file("hpsplit", tempdir);
			asprintf_safe(&cmd, "gunzip -cd %s > %s", infn, tempfn);
			logmsg("Running: \"%s\"\n", cmd);
			rtn = run_command_get_outputs(cmd, NULL, NULL);
			if (rtn) {
				ERROR("Failed to run command: \"%s\"", cmd);
				exit(-1);
			}
			free(cmd);
			infn = tempfn;
		}

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

		fitstable_use_buffered_reading(intable, 2*sizeof(double), 1000);

		R = fitstable_row_size(intable);
		rowbuf = buffered_read_new(R, 1000, NR, refill_rowbuffer, intable);

		if (fitstable_read_extension(intable, 1)) {
			ERROR("Failed to find RA and DEC columns (called \"%s\" and \"%s\" in the FITS file)", racol, deccol);
			exit(-1);
		}

		for (r=0; r<NR; r++) {
			int hp = -1;
			double ra, dec;
			int j;
			double* rd;
			void* rowdata;
			void* rdata;

			if (r && ((r % 100000) == 0)) {
			  logmsg("Reading row %i of %i\n", r, NR);
			}

			//printf("reading RA,Dec for row %i\n", r);
			rd = fitstable_next_struct(intable);
			ra = rd[0];
			dec = rd[1];

			logverb("row %i: ra,dec %g,%g\n", r, ra, dec);
			if (margin == 0) {
				hp = radecdegtohealpix(ra, dec, nside);
				logverb("  --> healpix %i\n", hp);
			} else {

				double xyz[3];
				anbool gotit = FALSE;
				double d2;
				if (!hps)
					hps = il_new(4);
				radecdeg2xyzarr(ra, dec, xyz);
				for (j=0; j<NHP; j++) {
					d2 = distsq(xyz, mincaps[j].xyz, 3);
					if (d2 <= mincaps[j].r2) {
						logverb("  -> in mincap %i  (dist %g vs %g)\n", j, sqrt(d2), sqrt(mincaps[j].r2));
						il_append(hps, j);
						gotit = TRUE;
						break;
					}
				}
				if (!gotit) {
					for (j=0; j<NHP; j++) {
						d2 = distsq(xyz, maxcaps[j].xyz, 3);
						if (d2 <= maxcaps[j].r2) {
							logverb("  -> in maxcap %i  (dist %g vs %g)\n", j, sqrt(d2), sqrt(maxcaps[j].r2));
							if (healpix_within_range_of_xyz(j, nside, xyz, margin)) {
								logverb("  -> and within range.\n");
								il_append(hps, j);
							}
						}
					}
				}

				//hps = healpix_rangesearch_radec(ra, dec, margin, nside, hps);

				logverb("  --> healpixes: [");
				for (j=0; j<il_size(hps); j++)
					logverb(" %i", il_get(hps, j));
				logverb(" ]\n");
			}

			//printf("Reading rowdata for row %i\n", r);
			rowdata = buffered_read(rowbuf);
			assert(rowdata);


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

					// MEMLEAK the output filename.  You'll live.
					asprintf_safe(&outfn, outfnpat, hp);
					logmsg("Opening output file \"%s\"...\n", outfn);
					out = fitstable_open_for_writing(outfn);
					if (!out) {
						ERROR("Failed to open output table \"%s\"", outfn);
						exit(-1);
					}
					// Set the output table structure.
					if (cols) {
					  fitstable_add_fits_columns_as_struct3(intable, out, cols, 0);
					} else
						fitstable_add_fits_columns_as_struct2(intable, out);

					if (backref) {
						tfits_type i16type;
						tfits_type i32type;
						// R = fitstable_row_size(intable);
						int off = R;
						i16type = fitscolumn_i16_type();
						i32type = fitscolumn_i32_type();
						fitstable_add_read_column_struct(out, i16type, 1, off,
														 i16type, "backref_file", TRUE);
						off += sizeof(int16_t);
						fitstable_add_read_column_struct(out, i32type, 1, off,
														 i32type, "backref_index", TRUE);
					}

					//printf("Output table:\n");
					//fitstable_print_columns(out);

					if (fitstable_write_primary_header(out) ||
						fitstable_write_header(out)) {
						ERROR("Failed to write output file headers for \"%s\"", outfn);
						exit(-1);
					}
					outtables[hp] = out;
				}

				if (backref) {
					int16_t brfile;
					int32_t brind;
					if (!padrowdata) {
						padrowdata = malloc(R + sizeof(int16_t) + sizeof(int32_t));
						assert(padrowdata);
					}
					// convert to FITS endian
					brfile = htons(i);
					brind  = htonl(r);
					// add backref data to rowdata
					memcpy(padrowdata, rowdata, R);
					memcpy(padrowdata + R, &brfile, sizeof(int16_t));
					memcpy(padrowdata + R + sizeof(int16_t), &brind, sizeof(int32_t));
					rdata = padrowdata;
				} else {
					rdata = rowdata;
				}

				if (cols) {
				  if (fitstable_write_struct_noflip(outtables[hp], rdata)) {
				    ERROR("Failed to copy a row of data from input table \"%s\" to output healpix %i", infn, hp);
				  }
				} else {
				  if (fitstable_write_row_data(outtables[hp], rdata)) {
				    ERROR("Failed to copy a row of data from input table \"%s\" to output healpix %i", infn, hp);
				  }
				}

				if (!hps)
					break;
			}
			if (hps)
				il_remove_all(hps);

		}
		buffered_read_free(rowbuf);
		// wack... buffered_read_free() just frees its internal buffer,
		// not the "rowbuf" struct itself.
		// who wrote this crazy code?  Oh, me of 5 years ago.  Jerk.
		free(rowbuf);

		fitstable_close(intable);
		il_free(hps);

		if (tempfn) {
			logverb("Removing temp file %s\n", tempfn);
			if (unlink(tempfn)) {
				SYSERROR("Failed to unlink() temp file \"%s\"", tempfn);
			}
			tempfn = NULL;
		}

		// fix headers so that the files are valid at this point.
		for (ii=0; ii<NHP; ii++) {
		  if (!outtables[ii])
		    continue;
		  off_t offset = ftello(outtables[ii]->fid);
		  if (fitstable_fix_header(outtables[ii])) {
		    ERROR("Failed to fix header for healpix %i after reading input file \"%s\"", ii, originfn);
		    exit(-1);
		  }
		  fseeko(outtables[ii]->fid, offset, SEEK_SET);
		}

		if (padrowdata) {
			free(padrowdata);
			padrowdata = NULL;
		}

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
	sl_free2(cols);

	free(mincaps);
	free(maxcaps);

    return 0;
}



