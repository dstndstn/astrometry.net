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
    fitsfile *fptr;  /* FITS file pointer */
    int status = 0;  /* CFITSIO status value MUST be initialized to zero! */
    int hdutype, naxis, ii;
    long naxes[2], totpix=0, fpixel[2];
    double *pix, sum = 0., meanval = 0., minval = 1.E33, maxval = -1.E33;

    if (argc != 2) { 
        printf("Usage: imstat image \n");
        printf("\n");
        printf("Compute statistics of pixels in the input image\n");
        printf("\n");
        printf("Examples: \n");
        printf("  imarith image.fits                    - the whole image\n");
        printf("  imarith 'image.fits[200:210,300:310]' - image section\n");
        printf("  imarith 'table.fits+1[bin (X,Y) = 4]' - image constructed\n");
        printf("     from X and Y columns of a table, with 4-pixel bin size\n");
        return(0);
    }

    if ( !fits_open_file(&fptr, argv[1], READONLY, &status) )
        {
            if (fits_get_hdu_type(fptr, &hdutype, &status) || hdutype != IMAGE_HDU) { 
                printf("Error: this program only works on images, not tables\n");
                return(1);
            }

            fits_get_img_dim(fptr, &naxis, &status);
            fits_get_img_size(fptr, 2, naxes, &status);

            if (status || naxis != 2) { 
                printf("Error: NAXIS = %d.  Only 2-D images are supported.\n", naxis);
                return(1);
            }

            pix = (double *) malloc(naxes[0] * sizeof(double)); /* memory for 1 row */

            if (pix == NULL) {
                printf("Memory allocation error\n");
                return(1);
            }

            totpix = naxes[0] * naxes[1];
            fpixel[0] = 1;  /* read starting with first pixel in each row */

            /* process image one row at a time; increment row # in each loop */
            for (fpixel[1] = 1; fpixel[1] <= naxes[1]; fpixel[1]++)
                {  
                    /* give starting pixel coordinate and number of pixels to read */
                    if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0],0, pix,0, &status))
                        break;   /* jump out of loop on error */

                    for (ii = 0; ii < naxes[0]; ii++) {
                        sum += pix[ii];                      /* accumlate sum */
                        if (pix[ii] < minval) minval = pix[ii];  /* find min and  */
                        if (pix[ii] > maxval) maxval = pix[ii];  /* max values    */
                    }
                }
      
            free(pix);
            fits_close_file(fptr, &status);
        }

    if (status)  {
        fits_report_error(stderr, status); /* print any error message */
    } else {
        if (totpix > 0) meanval = sum / totpix;

        printf("Statistics of %ld x %ld  image\n",
               naxes[0], naxes[1]);
        printf("  sum of pixels = %g\n", sum);
        printf("  mean value    = %g\n", meanval);
        printf("  minimum value = %g\n", minval);
        printf("  maximum value = %g\n", maxval);
    }

    return(status);
}

