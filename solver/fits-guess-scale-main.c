/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "fits-guess-scale.h"
#include "fitsioutils.h"
#include "log.h"

static char* OPTIONS = "hv";

static void printHelp(char* progname) {
    printf("%s  <FITS-file>\n\n", progname);
}


int main(int argc, char *argv[]) {
    char* progname = argv[0];
    int argchar;
    char* infn;

    sl* methods = NULL;
    dl* scales = NULL;
    int i;
    int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        case 'v':
            loglvl++;
            break;
        default:
            return -1;
        }

    if (optind != (argc - 1)) {
        printHelp(progname);
        exit(-1);
    }
    infn = argv[optind];

    log_init(loglvl);

    fits_use_error_system();

    if (fits_guess_scale(infn, &methods, &scales))
        exit(-1);

    for (i=0; i<sl_size(methods); i++) {
        printf("scale %s %g\n", sl_get(methods, i), dl_get(scales, i));
    }

    sl_free2(methods);
    dl_free(scales);

    return 0;
}

