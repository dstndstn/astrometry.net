/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "log.h"
#include "errors.h"
#include "sip.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "mathutil.h"
#include "boilerplate.h"

const char* OPTIONS = "hve:x:y:X:Y:s:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "\nUsage: %s [options] <input-wcs-file> <output-wcs-file>\n"
            "\n"
            "    [-e <extension>]  Read from given HDU (default 0 = primary)\n"
            "    [-x <x-lo>] (default 1)\n"
            "    [-X <x-hi>] (default image W)\n"
            "    [-y <y-lo>] (default 1)\n"
            "    [-Y <y-hi>] (default image H)\n"
            "    [-s <scale>] make output image this factor bigger (default 1)\n"
            "\n", progname);
}


int main(int argc, char** args) {
    int argchar;
    int loglvl = LOG_MSG;
    char* progname = args[0];
    char** myargs;
    int nmyargs = 0;
    int ext = 0;
    tan_t wcsin;
    tan_t wcsout;
    double xlo = 1;
    double xhi = -1;
    double ylo = 1;
    double yhi = -1;
    double scale = 1.0;
    double W, H;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1) {
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 'e':
            ext = atoi(optarg);
            break;
        case 's':
            scale = atof(optarg);
            break;
        case 'x':
            xlo = atof(optarg);
            break;
        case 'X':
            xhi = atof(optarg);
            break;
        case 'y':
            ylo = atof(optarg);
            break;
        case 'Y':
            yhi = atof(optarg);
            break;
        case 'h':
        default:
            printHelp(progname);
            exit(-1);
        }
    }
    if (optind < argc) {
        nmyargs = argc - optind;
        myargs = args + optind;
    }
    if (nmyargs != 2) {
        printHelp(progname);
        exit(-1);
    }

    log_init(loglvl);

    if (!tan_read_header_file_ext(myargs[0], ext, &wcsin)) {
        ERROR("failed to read WCS header from file %s, extension %i", myargs[0], ext);
        return -1;
    }

    W = wcsin.imagew;
    H = wcsin.imageh;
    if ((W == 0.0) || (H == 0.0)) {
        ERROR("failed to find IMAGE{W,H} in input WCS file");
        return -1;
    }

    if (xhi == -1)
        xhi = W;
    if (yhi == -1)
        yhi = H;

    logmsg("Cropping image to x=%g,%g, y=%g,%g and scaling by %g\n",
           xlo, xhi, ylo, yhi, scale);

    tan_transform(&wcsin, &wcsout, xlo, xhi, ylo, yhi, scale);


    if (tan_write_to_file(&wcsout, myargs[1])) {
        ERROR("Error writing output");
        exit(-1);
    }

    return 0;
}
