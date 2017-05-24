/*
  This file was downloaded from the CFITSIO utilities web page:
    http://heasarc.gsfc.nasa.gov/docs/software/fitsio/cexamples.html

  That page contains this text:
    You may freely modify, reuse, and redistribute these programs as you wish.

  We assume it was originally written by the CFITSIO authors (primarily William
  D. Pence).

  We (the Astrometry.net team) have modified it slightly.
  # Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <string.h>
#include <stdio.h>
#include "fitsio.h"

void print_usage() {
    printf("Usage:  tablist [-r] [-w 100] filename[ext][col filter][row filter] \n");
    printf("\n");
    printf("List the contents of a FITS table \n");
    printf(" -r: Don't output column headers or row numbers \n");
    printf(" -w 100: Print up to 100 characters per line; default 80\n");
    printf("\n");
    printf("Examples: \n");
    printf("  tablist tab.fits[GTI]           - list the GTI extension\n");
    printf("  tablist tab.fits[1][#row < 101] - list first 100 rows\n");
    printf("  tablist tab.fits[1][col X;Y]    - list X and Y cols only\n");
    printf("  tablist tab.fits[1][col -PI]    - list all but the PI col\n");
    printf("  tablist tab.fits[1][col -PI][#row < 101]  - combined case\n");
    printf("\n");
    printf("Display formats can be modified with the TDISPn keywords.\n");
}

int main(int argc, char *argv[])
{
    fitsfile *fptr;      /* FITS file pointer, defined in fitsio.h */
    char *val, value[1000], nullstr[]="*";
    char keyword[FLEN_KEYWORD], colname[FLEN_VALUE];
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */
    int hdunum, hdutype = ANY_HDU, ncols, ii, anynul, dispwidth[1000];
    long nelements[1000];
    int firstcol, lastcol = 1, linewidth;
    int max_linewidth = 80;
    int elem, firstelem, lastelem = 0, nelems;
    long jj, nrows, kk;
	int quiet = 0;
    int i;
	
    if (argc < 2) {
        print_usage();
        return 0;
    }

    if (fits_open_file(&fptr, argv[argc-1], READONLY, &status)) {
		if (status) fits_report_error(stderr, status); /* print any error message */
		return status;
	}
	if ( fits_get_hdu_num(fptr, &hdunum) == 1 )
		/* This is the primary array;  try to move to the */
		/* first extension and see if it is a table */
		fits_movabs_hdu(fptr, 2, &hdutype, &status);
	else
		fits_get_hdu_type(fptr, &hdutype, &status); /* Get the HDU type */

	if (!(hdutype == ASCII_TBL || hdutype == BINARY_TBL)) {
		printf("Error: this program only displays tables, not images\n");
		return -1;
	}

    for (i=1; i<argc-1; i++) {
        if (argv[i][0] != '-') {
            printf("Error: failed to parse command line; expected flag but got \"%s\"\n", argv[i]);
            return -1;
        }
        char flag = argv[i][1];
        if (flag == 'r') {
            quiet = 1;
        } else if (flag == 'w') {
            max_linewidth = atoi(argv[i+1]);
            i++;
        } else {
            printf("Error: failed to parse command line flag -%c\n", flag);
            return -1;
        }
    }

	fits_get_num_rows(fptr, &nrows, &status);
	fits_get_num_cols(fptr, &ncols, &status);

	for (jj=1; jj<=ncols; jj++)
		fits_get_coltype(fptr, jj, NULL, &nelements[jj], NULL, &status);
	//printf("nelements[%i] = %i.\n", (int)jj, (int)nelements[jj]);

	/* find the number of columns that will fit within max_linewidth
     characters */
	for (;;) {
		int breakout = 0;
		linewidth = 0;
		/* go on to the next element in the current column. */
		/* (if such an element does not exist, the inner 'for' loop
		   does not get run and we skip to the next column.) */
		firstcol = lastcol;
		firstelem = lastelem + 1;
		elem = firstelem;

		for (lastcol = firstcol; lastcol <= ncols; lastcol++) {
			int typecode;
			fits_get_col_display_width(fptr, lastcol, &dispwidth[lastcol], &status);
			fits_get_coltype(fptr, lastcol, &typecode, NULL, NULL, &status);
			typecode = abs(typecode);
			if (typecode == TBIT)
				nelements[lastcol] = (nelements[lastcol] + 7)/8;
			else if (typecode == TSTRING)
				nelements[lastcol] = 1;
			nelems = nelements[lastcol];
			for (lastelem = elem; lastelem <= nelems; lastelem++) {
				int nextwidth = linewidth + dispwidth[lastcol] + 1;
				if (nextwidth > max_linewidth) {
					breakout = 1;
					break;
				}
				linewidth = nextwidth;
			}
			if (breakout)
				break;
			/* start at the first element of the next column. */
			elem = 1;
		}

		/*
		  printf("ncols %i, linewidth %i, firstcol %i, lastcol %i, firstelem %i, lastelem %i, "
		  "nelements[lastcol-1] %i\n",
		  ncols, linewidth, firstcol, lastcol, firstelem, lastelem,
		  (int)nelements[lastcol-1]);
		*/

		/* if we exited the loop naturally, (not via break) then include all columns. */
		if (!breakout) {
			lastcol = ncols;
			lastelem = nelements[lastcol];
		}

		if (linewidth == 0)
			break;

		/* print column names as column headers */
		if (!quiet) {
			printf("\n    ");
			for (ii = firstcol; ii <= lastcol; ii++) {
				int maxelem;
				fits_make_keyn("TTYPE", ii, keyword, &status);
				fits_read_key(fptr, TSTRING, keyword, colname, NULL, &status);
				colname[dispwidth[ii]] = '\0';  /* truncate long names */
				kk = ((ii == firstcol) ? firstelem : 1);
				maxelem = ((ii == lastcol) ? lastelem : nelements[ii]);

				for (; kk <= maxelem; kk++) {
					if (kk > 1) {
						char buf[32];
						int len = snprintf(buf, 32, "(%li)", kk);
						printf("%*.*s%s ",dispwidth[ii]-len, dispwidth[ii]-len, colname, buf);
					} else
						printf("%*s ",dispwidth[ii], colname);
				}
			}
			printf("\n");  /* terminate header line */
		}

		/* print each column, row by row (there are faster ways to do this) */
		val = value;
		for (jj = 1; jj <= nrows && !status; jj++) {
			if (!quiet) 
				printf("%4d ", (int)jj);
			for (ii = firstcol; ii <= lastcol; ii++)
				{
					kk = ((ii == firstcol) ? firstelem : 1);
					nelems = ((ii == lastcol) ? lastelem : nelements[ii]);
					for (; kk <= nelems; kk++)
						{
							/* read value as a string, regardless of intrinsic datatype */
							if (fits_read_col_str (fptr,ii,jj,kk, 1, nullstr,
												   &val, &anynul, &status) )
								break;  /* jump out of loop on error */
							printf("%-*s ",dispwidth[ii], value);
						}
				}
			printf("\n");
		}

		if (!breakout)
			break;
	}
	fits_close_file(fptr, &status);

	if (status) fits_report_error(stderr, status); /* print any error message */
	return(status);
}
