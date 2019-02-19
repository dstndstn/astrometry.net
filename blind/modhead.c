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
    fitsfile *fptr;         /* FITS file pointer, defined in fitsio.h */
    char card[FLEN_CARD], newcard[FLEN_CARD];
    char oldvalue[FLEN_VALUE], comment[FLEN_COMMENT];
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */
    int iomode, keytype;

    if (argc == 3) 
        iomode = READONLY;
    else if ((argc == 4) || (argc == 5))
        iomode = READWRITE;
    else {
        printf("Usage:  modhead filename[ext] keyword newvalue [newcomment]\n");
        printf("\n");
        printf("Write or modify the value of a header keyword.\n");
        printf("If 'newvalue' is not specified then just print \n");
        printf("the current value. \n");
        printf("\n");
        printf("Examples: \n");
        printf("  modhead file.fits dec      - list the DEC keyword \n");
        printf("  modhead file.fits dec 30.0 - set DEC = 30.0 \n");
        printf("  modhead file.fits dec 30.0 \"The decline of civilization\" - set DEC = 30.0 and add a comment \n");
        return(0);
    }

    if (!fits_open_file(&fptr, argv[1], iomode, &status))
        {
            if (fits_read_card(fptr,argv[2], card, &status))
                {
                    printf("Keyword does not exist\n");
                    card[0] = '\0';
                    comment[0] = '\0';
                    status = 0;  /* reset status after error */
                }
            else
                printf("%s\n",card);

            if ((argc == 4) || (argc == 5)) /* write or overwrite the keyword */
                {
                    /* check if this is a protected keyword that must not be changed */
                    /*
                     if (*card && fits_get_keyclass(card) == TYP_STRUC_KEY)
                     {
                     printf("Protected keyword cannot be modified.\n");
                     }
                     else
                     */
                    {
                        /* get the comment string */
			if (*card)
                            fits_parse_value(card, oldvalue, comment, &status);

                        /* construct template for new keyword */
                        strcpy(newcard, argv[2]);     /* copy keyword name */
                        strcat(newcard, " = ");       /* '=' value delimiter */
                        strcat(newcard, argv[3]);     /* new value */
			if (argc == 5) {
                            strcat(newcard, " / ");  /* comment delimiter */
                            strcat(newcard, argv[4]);     /* append the comment */
			} else if (*comment) {
                            /* old comment */
                            strcat(newcard, " / ");  /* comment delimiter */
                            strcat(newcard, comment);     /* append the comment */
                        }

                        /* reformat the keyword string to conform to FITS rules */
                        fits_parse_template(newcard, card, &keytype, &status);

                        /* overwrite the keyword with the new value */
                        fits_update_card(fptr, argv[2], card, &status);

                        printf("Keyword has been changed to:\n");
                        printf("%s\n",card);
                    }
                }
            fits_close_file(fptr, &status);
        }

    /* if error occured, print out error message */
    if (status) fits_report_error(stderr, status);
    return(status);
}

