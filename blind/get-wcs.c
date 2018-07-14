/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 Reads a FITS file, tries to interpret a WCS header and writes out a TAN
 interpretation of it.
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include "sip.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "ioutils.h"

static char* OPTIONS = "ho:e:t";

static void printHelp(char* progname) {
    printf("%s <input-file>\n"
           "   [-e <extension>]\n"
           "   [-o <output-file>]\n"
           "   [-t]: force TAN (not SIP)\n"
           "\n", progname);
}


int main(int argc, char *argv[]) {
    char* progname = argv[0];
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    sip_t* wcs;
    int ext = 0;
    anbool forcetan = FALSE;
    tan_t* tan;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        case 'e':
            ext = atoi(optarg);
            break;
        case 'o':
            outfn = optarg;
            break;
        case 't':
            forcetan = TRUE;
            break;
        default:
            return -1;
        }

    if (optind != (argc - 1)) {
        printHelp(progname);
        exit(-1);
    }
    infn = argv[optind];

    log_init(LOG_MSG);
    fits_use_error_system();
    errors_log_to(stderr);

    wcs = sip_read_tan_or_sip_header_file_ext(infn, ext, NULL, forcetan);
    if (!wcs) {
        ERROR("Failed to parse WCS header");
        exit(-1);
    }

    tan = &(wcs->wcstan);

    printf("crval1 %.12g\n", tan->crval[0]);
    printf("crval2 %.12g\n", tan->crval[1]);
    printf("crpix1 %g\n", tan->crpix[0]);
    printf("crpix2 %g\n", tan->crpix[1]);
    printf("cd11 %.12g\n", tan->cd[0][0]);
    printf("cd12 %.12g\n", tan->cd[0][1]);
    printf("cd21 %.12g\n", tan->cd[1][0]);
    printf("cd22 %.12g\n", tan->cd[1][1]);

    if (wcs->a_order) {
        printf("a_order %i\n", wcs->a_order);
        printf("b_order %i\n", wcs->b_order);
        // FIXME -- print the coefficients!
    }

    if (outfn) {
        FILE* fout;
        anbool tostdout;
        tostdout = streq(outfn, "-");
        if (tostdout)
            fout = stdout;
        else {
            fout = fopen(outfn, "wb");
            if (!fout) {
                SYSERROR("Failed to open output file %s", outfn);
                exit(-1);
            }
        }

        if (wcs->a_order) {
            if (sip_write_to(wcs, fout)) {
                ERROR("Failed to write SIP header to file \"%s\"", outfn);
                exit(-1);
            }
        } else {
            if (tan_write_to(&(wcs->wcstan), fout)) {
                ERROR("Failed to write TAN header to file \"%s\"", outfn);
                exit(-1);
            }
        }

        if (!tostdout) {
            if (fclose(fout)) {
                SYSERROR("Failed to close output file \"%s\"", outfn);
                exit(-1);
            }
        }
    }

    return 0;
}
