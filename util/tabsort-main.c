/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "tabsort.h"
#include "fitsioutils.h"

static const char* OPTIONS = "hd";

static void printHelp(char* progname) {
    printf("%s  [options]  <column-name> <input-file> <output-file>\n"
           "  options include:\n"
           "      [-d]: sort in descending order (default, ascending)\n",
           progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    char* colname = NULL;
    char* progname = argv[0];
    anbool descending = FALSE;

    while ((argchar = getopt(argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'd':
            descending = TRUE;
            break;
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (optind != argc-3) {
        printHelp(progname);
        exit(-1);
    }

    colname = argv[optind  ];
    infn    = argv[optind+1];
    outfn   = argv[optind+2];

    fits_use_error_system();

    return tabsort(infn, outfn, colname, descending);
}

