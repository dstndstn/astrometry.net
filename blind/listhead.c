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
    char card[FLEN_CARD];   /* Standard string lengths defined in fitsio.h */
    int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    int single = 0, hdupos, nkeys, ii;

    if (argc != 2) {
      printf("Usage:  listhead filename[ext] \n");
      printf("\n");
      printf("List the FITS header keywords in a single extension, or, if \n");
      printf("ext is not given, list the keywords in all the extensions. \n");
      printf("\n");
      printf("Examples: \n");
      printf("   listhead file.fits      - list every header in the file \n");
      printf("   listhead file.fits[0]   - list primary array header \n");
      printf("   listhead file.fits[2]   - list header of 2nd extension \n");
      printf("   listhead file.fits+2    - same as above \n");
      printf("   listhead file.fits[GTI] - list header of GTI extension\n");
      printf("\n");
      printf("Note that it may be necessary to enclose the input file\n");
      printf("name in single quote characters on the Unix command line.\n");
      return(0);
    }

    if (!fits_open_file(&fptr, argv[1], READONLY, &status))
    {
      fits_get_hdu_num(fptr, &hdupos);  /* Get the current HDU position */

      /* List only a single header if a specific extension was given */ 
      if (hdupos != 1 || strchr(argv[1], '[')) single = 1;

      for (; !status; hdupos++)  /* Main loop through each extension */
      {
        fits_get_hdrspace(fptr, &nkeys, NULL, &status); /* get # of keywords */

        printf("Header listing for HDU #%d:\n", hdupos);

        for (ii = 1; ii <= nkeys; ii++) { /* Read and print each keywords */

           if (fits_read_record(fptr, ii, card, &status))break;
           printf("%s\n", card);
        }
        printf("END\n\n");  /* terminate listing with END */

        if (single) break;  /* quit if only listing a single header */

        fits_movrel_hdu(fptr, 1, NULL, &status);  /* try to move to next HDU */
      }

      if (status == END_OF_FILE)  status = 0; /* Reset after normal error */

      fits_close_file(fptr, &status);
    }

    if (status) fits_report_error(stderr, status); /* print any error message */
    return(status);
}

