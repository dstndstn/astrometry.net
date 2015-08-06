/*
  This file was downloaded from the CFITSIO utilities web page:
    http://heasarc.gsfc.nasa.gov/docs/software/fitsio/cexamples.html

  That page contains this text:
    You may freely modify, reuse, and redistribute these programs as you wish.

  We assume it was originally written by the CFITSIO authors (primarily William
  D. Pence).

  We (the Astrometry.net team) have modified it slightly, and we hereby place
  our modifications under the GPLv2.
*/
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

#include <string.h>
#include <stdio.h>
#include "fitsio.h"

int main(int argc, char *argv[])
{
    fitsfile *fptr;         /* FITS file pointer, defined in fitsio.h */
    char keyname[FLEN_KEYWORD], colname[FLEN_VALUE], coltype[FLEN_VALUE];
    int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    int single = 0, hdupos, hdutype, bitpix, naxis, ncols, ii;
    long naxes[10], nrows;

    if (argc != 2) {
      printf("Usage:  liststruc filename[ext] \n");
      printf("\n");
      printf("List the structure of a single extension, or, if ext is \n");
      printf("not given, list the structure of the entire FITS file.  \n");
      printf("\n");
      printf("Note that it may be necessary to enclose the input file\n");
      printf("name in single quote characters on the Unix command line.\n");
      return(0);
    }

    if (!fits_open_file(&fptr, argv[1], READONLY, &status))
    {
      fits_get_hdu_num(fptr, &hdupos);  /* Get the current HDU position */

      /* List only a single structure if a specific extension was given */ 
      if (strchr(argv[1], '[') || strchr(argv[1], '+')) single++;

      for (; !status; hdupos++)   /* Main loop for each HDU */
      {
        fits_get_hdu_type(fptr, &hdutype, &status);  /* Get the HDU type */

        printf("\nHDU #%d  ", hdupos);
        if (hdutype == IMAGE_HDU)   /* primary array or image HDU */
        {
          fits_get_img_param(fptr, 10, &bitpix, &naxis, naxes, &status);

          printf("Array:  NAXIS = %d,  BITPIX = %d\n", naxis, bitpix);
          for (ii = 0; ii < naxis; ii++)
            printf("   NAXIS%d = %ld\n",ii+1, naxes[ii]);  
        }
        else  /* a table HDU */
        {
          fits_get_num_rows(fptr, &nrows, &status);
          fits_get_num_cols(fptr, &ncols, &status);

          if (hdutype == ASCII_TBL)
            printf("ASCII Table:  ");
          else
            printf("Binary Table:  ");

          printf("%d columns x %ld rows\n", ncols, nrows);
          printf(" COL NAME             FORMAT\n");

          for (ii = 1; ii <= ncols; ii++)
          {
            fits_make_keyn("TTYPE", ii, keyname, &status); /* make keyword */
            fits_read_key(fptr, TSTRING, keyname, colname, NULL, &status);
            fits_make_keyn("TFORM", ii, keyname, &status); /* make keyword */
            fits_read_key(fptr, TSTRING, keyname, coltype, NULL, &status);

            printf(" %3d %-16s %-16s\n", ii, colname, coltype);
          }
        }

        if (single)break;  /* quit if only listing a single HDU */

        fits_movrel_hdu(fptr, 1, NULL, &status);  /* try move to next ext */
      }

      if (status == END_OF_FILE) status = 0; /* Reset normal error */
      fits_close_file(fptr, &status);
    }

    if (status) fits_report_error(stderr, status); /* print any error message */
    return(status);
}

