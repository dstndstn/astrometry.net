/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "fitsioutils.h"
#include "errors.h"
#include "boilerplate.h"
#include "ioutils.h"
#include "merge-index.h"

#define OPTIONS "hq:c:s:o:"

static void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -q <input-quad-filename>\n"
           "   -c <input-code-kdtree-filename>\n"
           "   -s <input-star-kdtree-filename>\n"
           "   -o <output-index-filename>\n"
           "\n", progname);
}


int main(int argc, char **args) {
    int argchar;
    char* progname = args[0];
    char* quadfn = NULL;
    char* codefn = NULL;
    char* starfn = NULL;
    char* outfn = NULL;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'q':
            quadfn = optarg;
            break;
        case 'c':
            codefn = optarg;
            break;
        case 's':
            starfn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (!(quadfn && starfn && codefn && outfn)) {
        printHelp(progname);
        fprintf(stderr, "\nYou must specify all filenames (-q, -c, -s, -o)\n");
        exit(-1);
    }

    fits_use_error_system();

    if (merge_index_files(quadfn, codefn, starfn, outfn)) {
        exit(-1);
    }
    return 0;
}
