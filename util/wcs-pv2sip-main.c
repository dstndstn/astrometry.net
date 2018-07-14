/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <math.h>

#include "wcs-pv2sip.h"
#include "boilerplate.h"
#include "bl.h"
#include "log.h"
#include "fitsioutils.h"

const char* OPTIONS = "hve:sx:X:y:Y:a:W:H:to:S";

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <input-wcs> <output-wcs>\n"
           "   [-o <order>] SIP polynomial order to fit (default: 5)\n"
           "   [-e <extension>] FITS HDU number to read WCS from (default 0 = primary)\n"
           "   [-S]: do NOT do the wcs_shift thing\n"
           "   [-s]: treat input as Scamp .head file\n"
           "   [-t]: override the CTYPE* cards in the WCS header, and assume they are TAN.\n"
           "   [-v]: +verboseness\n"
           " Set the IMAGEW, IMAGEH in the output file:\n"
           "   [-W <int>]\n"
           "   [-H <int>]\n"
           " Set the pixel values used to compute the distortion polynomials with:\n"
           "   [-x <x-low>] (default: 1)\n"
           "   [-y <y-low>] (default: 1)\n"
           "   [-X <x-high>] (default: image width)\n"
           "   [-Y <y-high>] (default: image width)\n"
           "   [-a <step-size>] (default: closest to 100 yielding whole number of steps)\n"
           "\n", progname);
}


int main(int argc, char** args) {
    int loglvl = LOG_MSG;
    char** myargs;
    int nargs;
    int c;
    int order = 5;

    char* wcsinfn = NULL;
    char* wcsoutfn = NULL;
    int ext = 0;
    anbool scamp = FALSE;
    double xlo = 0;
    double xhi = 0;
    double stepsize = 0;
    double ylo = 0;
    double yhi = 0;
    anbool forcetan = FALSE;
    int W, H;
    int doshift = 1;

    W = H = 0;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'S':
            doshift = 0;
            break;
        case 't':
            forcetan = TRUE;
            break;
        case 'o':
            order = atoi(optarg);
            break;
        case 'W':
            W = atoi(optarg);
            break;
        case 'H':
            H = atoi(optarg);
            break;
        case 'x':
            xlo = atof(optarg);
            break;
        case 'X':
            xhi = atof(optarg);
            break;
        case 'a':
            stepsize = atof(optarg);
            break;
        case 'y':
            ylo = atof(optarg);
            break;
        case 'Y':
            yhi = atof(optarg);
            break;
        case 's':
            scamp = TRUE;
            break;
        case 'e':
            ext = atoi(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
        case 'h':
            print_help(args[0]);
            exit(0);
        }
    }
    nargs = argc - optind;
    myargs = args + optind;

    if (nargs != 2) {
        print_help(args[0]);
        exit(-1);
    }
    wcsinfn = myargs[0];
    wcsoutfn = myargs[1];

    log_init(loglvl);
    fits_use_error_system();

    logmsg("Reading WCS (with PV distortions) from %s, ext %i\n", wcsinfn, ext);
    logmsg("Writing WCS (with SIP distortions) to %s\n", wcsoutfn);

    if (wcs_pv2sip(wcsinfn, ext, wcsoutfn, scamp, NULL, 0,
                   stepsize, xlo, xhi, ylo, yhi, W, H,
                   order, forcetan, doshift)) {
        exit(-1);
    }

    return 0;
}
