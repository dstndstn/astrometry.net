/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

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

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include "fitsio.h"
#include "dimage.h"
#include "svn.h"

void printHelp(char* progname)
{
	fprintf(stderr, "%s usage:\n"
	        "   -i <input file>      file to tweak\n"
	        "   -m actually modify file\n"
	        , progname);
}

int main(int argc, char *argv[])
{
	fitsfile *fptr;         /* FITS file pointer, defined in fitsio.h */
	//char card[FLEN_CARD];   /* Standard string lengths defined in fitsio.h */
	int status = 0; // FIXME should have ostatus too
	int naxis;
	long naxisn[100];
	int fix = 0;
	int kk;
	char* infile = NULL;
	int argchar;
	int nhdus,hdutype;
	int broken;

	while ((argchar = getopt(argc, argv, "hmi:")) != -1)
		switch (argchar) {
		case 'h':
			printHelp(argv[0]);
			exit(0);
		case 'i':
			infile = optarg;
			break;
		case 'm':
			fix = 1;
			break;
		}

	if (!infile) {
		printHelp(argv[0]);
		exit( -1);
	}
	if (fits_open_file(&fptr, infile, fix ? READWRITE : READONLY, &status)) {
		fprintf(stderr, "Error reading file %s\n", infile);
		fits_report_error(stderr, status);
		exit(-1);
	}

	// Are there multiple HDU's?
	fits_get_num_hdus(fptr, &nhdus, &status);
	fprintf(stderr, "nhdus=%d\n", nhdus);

	if (fix) {
        char str[256];
		fits_write_history(fptr, 
			"Edited by Astrometry.net's fixaxis",
			&status);
        snprintf(str, sizeof(str), "SVN rev: %i", svn_revision());
		fits_write_history(fptr, str, &status);
		assert(!status);
        snprintf(str, sizeof(str), "SVN URL: %s", svn_url());
		fits_write_history(fptr, str, &status);
		assert(!status);
        snprintf(str, sizeof(str), "SVN date: %s", svn_date());
		fits_write_history(fptr, str, &status);
		assert(!status);
		fits_write_history(fptr, 
			"Visit us on the web at http://astrometry.net/",
			&status);
		assert(!status);
	}

//	int nimgs = 0;
	broken = 0; // Correct until proven otherwise!

	// Check axis on each one
	for (kk=1; kk <= nhdus; kk++) {
		int naxis_actual = 0;
		char keyname[10];

		fits_movabs_hdu(fptr, kk, &hdutype, &status);
		fits_get_hdu_type(fptr, &hdutype, &status);

		if (hdutype != IMAGE_HDU) 
			continue;

		fits_get_img_dim(fptr, &naxis, &status);
		if (status) {
			fits_report_error(stderr, status);
			exit( -1);
		}

		// Assume that if naxis is not zero, then it is correct.
		// FIXME prove this is true; but by XP dogma, will not fix
		// until we find an image where naxis !=0 and naxis is not
		// correct. Trac #234
		if (naxis != 0)
			continue;


		// Now find if and how many NAXISN keywords there are
		while(1) {
			snprintf(keyname, 10, "NAXIS%d", naxis_actual+1);
			fits_read_key(fptr, TINT, keyname, naxisn+naxis_actual,
					NULL, &status);
			if (status) {
				// No more.
				status = 0;
				break;
			}
			naxis_actual++;
		}
		if (naxis_actual == 0) {
			// Ok, it's real.
			continue;
		}

		broken++;
		fprintf(stderr,"broken: actual naxis %d \n", naxis_actual);

		if (fix) {
//			char comment[80];
			//fits_read_key(fptr, TINT, "NAXIS", &naxis, comment, &status);
			//fprintf(stderr, "%s\n", comment);
			//assert(!status);
			fits_update_key(fptr, TINT, "NAXIS", &naxis_actual,
					"set by fixaxis", &status);
			assert(!status);
		}
	}

	if (status == END_OF_FILE)
		status = 0; /* Reset after normal error */

	fits_close_file(fptr, &status);

	if (status)
		fits_report_error(stderr, status); /* print any error message */
	return (status);
}
