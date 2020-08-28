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

int main(int argc, char *argv[])
{
    fitsfile *infptr, *outfptr;  /* FITS file pointers */
    int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    int icol, incols, outcols, intype, outtype, check = 1;
    long inrep, outrep, width, inrows, outrows, ii, jj;
    unsigned char *buffer = 0;

    if (argc != 3) { 
        printf("Usage:  tabmerge infile1[ext][filter] outfile[ext]\n");
        printf("\n");
        printf("Merge 2 tables by copying all the rows from the 1st table\n");
        printf("into the 2nd table.  The  2 tables must have identical\n");
        printf("structure, with the same number of columns with the same\n");
        printf("datatypes.  This program modifies the output file in place,\n");
        printf("rather than creating a whole new output file.\n");
        printf("\n");
        printf("Examples: \n");
        printf("\n");
        printf("1. tabmerge intab.fit+1 outtab.fit+2\n");
        printf("\n");
        printf("    merge the table in the 1st extension of intab.fit with\n");
        printf("    the table in the 2nd extension of outtab.fit.\n");
        printf("\n");
        printf("2. tabmerge 'intab.fit+1[PI > 45]' outab.fits+2\n");
        printf("\n");
        printf("    Same as the 1st example, except only rows that have a PI\n");
        printf("    column value > 45 will be merged into the output table.\n");
        printf("\n");
        return(0);
    }

    /* open both input and output files and perform validity checks */
    if ( fits_open_file(&infptr,  argv[1], READONLY,  &status) ||
         fits_open_file(&outfptr, argv[2], READWRITE, &status) )
        printf(" Couldn't open both files\n");
        
    else if ( fits_get_hdu_type(infptr,  &intype,  &status) ||
              fits_get_hdu_type(outfptr, &outtype, &status) )
        printf("couldn't get the type of HDU for the files\n");

    else if (intype == IMAGE_HDU) {
        printf("The input HDU is an image, not a table\n");
        exit(-1);
    } else if (outtype == IMAGE_HDU) {
        printf("The output HDU is an image, not a table\n");
        exit(-1);
    } else if (outtype != intype)
        printf("Input and output HDUs are not the same type of table.\n");

    else if ( fits_get_num_cols(infptr,  &incols,  &status) ||
              fits_get_num_cols(outfptr, &outcols, &status) )
        printf("Couldn't get number of columns in the tables\n");

    else if ( incols != outcols )
        printf("Input and output HDUs don't have same # of columns.\n");        

    else if ( fits_read_key(infptr, TLONG, "NAXIS1", &width, NULL, &status) )
        printf("Couldn't get width of input table\n");

    else if (!(buffer = (unsigned char *) malloc(width)) )
        printf("memory allocation error\n");

    else if ( fits_get_num_rows(infptr,  &inrows,  &status) ||
              fits_get_num_rows(outfptr, &outrows, &status) )
        printf("Couldn't get the number of rows in the tables\n");

    else  {
        /* check that the corresponding columns have the same datatypes */
        for (icol = 1; icol <= incols; icol++) {
            fits_get_coltype(infptr,  icol, &intype,  &inrep,  NULL, &status);
            fits_get_coltype(outfptr, icol, &outtype, &outrep, NULL, &status);
            if (intype != outtype || inrep != outrep) {
                printf("Column %d is not the same in both tables\n", icol);
                check = 0;
            }
        }

        if (check && !status) 
            {
                /* insert 'inrows' empty rows at the end of the output table */
                fits_insert_rows(outfptr, outrows, inrows, &status);

                for (ii = 1, jj = outrows +1; ii <= inrows; ii++, jj++)
                    {
                        /* read row from input and write it to the output table */
                        fits_read_tblbytes( infptr,  ii, 1, width, buffer, &status);
                        fits_write_tblbytes(outfptr, jj, 1, width, buffer, &status);
                        if (status)break;  /* jump out of loop if error occurred */
                    }

                /* all done; now free memory and close files */
                fits_close_file(outfptr, &status);
                fits_close_file(infptr,  &status); 
            }
    }

    if (buffer)
        free(buffer);

    if (status) fits_report_error(stderr, status); /* print any error message */
    return(status);
}

