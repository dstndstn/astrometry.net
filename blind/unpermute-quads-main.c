/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "unpermute-quads.h"
#include "boilerplate.h"

#define OPTIONS "hq:c:Q:C:"

static void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -q <input-quad-filename>\n"
           "   -c <input-code-kdtree-filename>\n"
           "   -Q <output-quad-filename>\n"
           "   -C <output-code-kdtree-filename>\n"
           "\n", progname);
}


int main(int argc, char **args) {
    int argchar;
    char* progname = args[0];
    char* quadinfn = NULL;
    char* quadoutfn = NULL;
    char* ckdtinfn = NULL;
    char* ckdtoutfn = NULL;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'q':
            quadinfn = optarg;
            break;
        case 'c':
            ckdtinfn = optarg;
            break;
        case 'Q':
            quadoutfn = optarg;
            break;
        case 'C':
            ckdtoutfn = optarg;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (!(quadinfn && quadoutfn && ckdtinfn && ckdtoutfn)) {
        printHelp(progname);
        fprintf(stderr, "\nYou must specify all filenames (-q, -c, -Q, -C)\n");
        exit(-1);
    }

    if (unpermute_quads_files(quadinfn, ckdtinfn, quadoutfn, ckdtoutfn,
                              args, argc)) {
        exit(-1);
    }
    return 0;
}
