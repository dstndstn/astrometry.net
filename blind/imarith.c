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
    fitsfile *afptr, *bfptr, *outfptr;  /* FITS file pointers */
    int status = 0;  /* CFITSIO status value MUST be initialized to zero! */
    int anaxis, bnaxis, check = 1, ii, op;
    long npixels = 1, firstpix[3] = {1,1,1};
    long anaxes[3] = {1,1,1}, bnaxes[3]={1,1,1};
    double *apix, *bpix;

    if (argc != 5) { 
        printf("Usage: imarith image1 image2 oper outimage \n");
        printf("\n");
        printf("Perform arithmetic operation on 2 input images\n");
        printf("creating a new output image.  Supported arithmetic\n");
        printf("operators are add, sub, mul, div (first character required\n");
        printf("\n");
        printf("Example: \n");
        printf("  imarith in1.fits in2.fits a out.fits - add the 2 files\n");
        return(0);
    }

    fits_open_file(&afptr, argv[1], READONLY, &status); /* open input images */
    fits_open_file(&bfptr, argv[2], READONLY, &status);

    fits_get_img_dim(afptr, &anaxis, &status);  /* read dimensions */
    fits_get_img_dim(bfptr, &bnaxis, &status);
    fits_get_img_size(afptr, 3, anaxes, &status);
    fits_get_img_size(bfptr, 3, bnaxes, &status);

    if (status) {
        fits_report_error(stderr, status); /* print error message */
        return(status);
    }

    if (anaxis > 3) {
        printf("Error: images with > 3 dimensions are not supported\n");
        check = 0;
    }
    /* check that the input 2 images have the same size */
    else if ( anaxes[0] != bnaxes[0] || 
              anaxes[1] != bnaxes[1] || 
              anaxes[2] != bnaxes[2] ) {
        printf("Error: input images don't have same size\n");
        check = 0;
    }

    if      (*argv[3] == 'a' || *argv[3] == 'A')
        op = 1;
    else if (*argv[3] == 's' || *argv[3] == 'S')
        op = 2;
    else if (*argv[3] == 'm' || *argv[3] == 'M')
        op = 3;
    else if (*argv[3] == 'd' || *argv[3] == 'D')
        op = 4;
    else {
        printf("Error: unknown arithmetic operator\n");
        check = 0;
    }

    /* create the new empty output file if the above checks are OK */
    if (check && !fits_create_file(&outfptr, argv[4], &status) )
        {
            /* copy all the header keywords from first image to new output file */
            fits_copy_header(afptr, outfptr, &status);

            npixels = anaxes[0];  /* no. of pixels to read in each row */

            apix = (double *) malloc(npixels * sizeof(double)); /* mem for 1 row */
            bpix = (double *) malloc(npixels * sizeof(double)); 

            if (apix == NULL || bpix == NULL) {
                printf("Memory allocation error\n");
                return(1);
            }

            /* loop over all planes of the cube (2D images have 1 plane) */
            for (firstpix[2] = 1; firstpix[2] <= anaxes[2]; firstpix[2]++)
                {
                    /* loop over all rows of the plane */
                    for (firstpix[1] = 1; firstpix[1] <= anaxes[1]; firstpix[1]++)
                        {
                            /* Read both images as doubles, regardless of actual datatype.  */
                            /* Give starting pixel coordinate and no. of pixels to read.    */
                            /* This version does not support undefined pixels in the image. */

                            if (fits_read_pix(afptr, TDOUBLE, firstpix, npixels, NULL, apix,
                                              NULL, &status)  ||         
                                fits_read_pix(bfptr, TDOUBLE, firstpix, npixels,  NULL, bpix,
                                              NULL, &status)  )        
                                break;   /* jump out of loop on error */

                            switch (op) {
                            case 1:         
                                for(ii=0; ii< npixels; ii++)
                                    apix[ii] += bpix[ii];
                                break;
                            case 2:
                                for(ii=0; ii< npixels; ii++)
                                    apix[ii] -= bpix[ii];
                                break;
                            case 3:
                                for(ii=0; ii< npixels; ii++)
                                    apix[ii] *= bpix[ii];
                                break;
                            case 4:
                                for(ii=0; ii< npixels; ii++) {
                                    if (bpix[ii] !=0.)
                                        apix[ii] /= bpix[ii];
                                    else
                                        apix[ii] = 0.;
                                }
                            }

                            fits_write_pix(outfptr, TDOUBLE, firstpix, npixels,
                                           apix, &status); /* write new values to output image */
                        }
                }    /* end of loop over planes */

            fits_close_file(outfptr, &status);
            free(apix);
            free(bpix);
        }

    fits_close_file(afptr, &status);
    fits_close_file(bfptr, &status);
 
    if (status) fits_report_error(stderr, status); /* print any error message */
    return(status);
}

