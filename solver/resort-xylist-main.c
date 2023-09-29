/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "an-bool.h"
#include "resort-xylist.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"

const char* OPTIONS = "hdf:b:v";

static void printHelp(char* progname) {
    printf("Usage:   %s  <input> <output>\n"
           "      -f <flux-column-name>  (default: FLUX) \n"
           "      -b <background-column-name>  (default: BACKGROUND)\n"
           "      [-d]: sort in descending order (default is ascending)\n"
           "      [-v]: add verboseness.\n"
           "\n", progname);
}


int main(int argc, char** args) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    char* progname = args[0];
    char* fluxcol = NULL;
    char* backcol = NULL;
    anbool ascending = TRUE;
    int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'f':
            fluxcol = optarg;
            break;
        case 'b':
            backcol = optarg;
            break;
        case 'd':
            ascending = FALSE;
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }
    log_init(loglvl);

    if (optind != argc-2) {
        printHelp(progname);
        exit(-1);
    }

    infn = args[optind];
    outfn = args[optind+1];

    fits_use_error_system();

    if (resort_xylist(infn, outfn, fluxcol, backcol, ascending)) {
        ERROR("Failed to re-sort xylist by FLUX and BACKGROUND columns \"%s\" and \"%s\"", fluxcol, backcol);
        exit(-1);
    }

    return 0;
}
