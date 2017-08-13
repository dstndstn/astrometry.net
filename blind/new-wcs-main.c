/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include "new-wcs.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"

static const char* OPTIONS = "hi:w:o:de:v";

static void printHelp(char* progname) {
    printf("%s    -i <input-file>\n"
           "      -w <WCS-file>\n"
           "      -o <output-file>\n"
           "      [-e <extension>]: (default: copy data from primary HDU)\n"
           "      [-d]: also copy the data segment\n"
           "      [-v]: +verbose\n"
           "\n",
           progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    char* wcsfn = NULL;
    char* progname = argv[0];
    anbool copydata = FALSE;
    int loglvl = LOG_MSG;
    int extension = 0;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 'e':
            extension = atoi(optarg);
            break;
        case 'i':
            infn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case 'w':
            wcsfn = optarg;
            break;
        case 'd':
            copydata = TRUE;
            break;
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (!infn || !outfn || !wcsfn) {
        printHelp(progname);
        exit(-1);
    }
    log_init(loglvl);
    fits_use_error_system();

    if (new_wcs(infn, extension, wcsfn, outfn, copydata)) {
        ERROR("new_wcs() failed");
        exit(-1);
    }
    return 0;
}
