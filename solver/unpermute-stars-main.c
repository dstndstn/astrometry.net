/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "kdtree.h"
#include "starutil.h"
#include "quadfile.h"
#include "fitsioutils.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "unpermute-stars.h"

static const char* OPTIONS = "hs:q:S:Q:wcv";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "    -s <input-star-kdtree-filename>\n"
           "    -q <input-quads-filename>\n"
           "    -S <output-star-kdtree-filename>\n"
           "    -Q <output-quads-filename>\n"
           "   [-w]: store sweep number in output star kdtree file.\n"
           "   [-c]: check values\n"
           "   [-v]: more verbose\n"
           "\n", progname);
}


int main(int argc, char **args) {
    int argchar;
    char* progname = args[0];
    char* quadinfn = NULL;
    char* skdtinfn = NULL;
    char* quadoutfn = NULL;
    char* skdtoutfn = NULL;
    anbool dosweeps = FALSE;
    anbool check = FALSE;
    int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'c':
            check = TRUE;
            break;
        case 'v':
            loglvl++;
            break;
        case 'q':
            quadinfn = optarg;
            break;
        case 'Q':
            quadoutfn = optarg;
            break;
        case 's':
            skdtinfn = optarg;
            break;
        case 'S':
            skdtoutfn = optarg;
            break;
        case 'w':
            dosweeps = TRUE;
            break;
        case '?':
            ERROR("Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    log_init(loglvl);

    if (!(quadinfn && quadoutfn && skdtinfn && skdtoutfn)) {
        printHelp(progname);
        ERROR("\nMust include all filenames (-q, -Q, -s, -S)\n");
        exit(-1);
    }

    if (unpermute_stars_files(skdtinfn, quadinfn, skdtoutfn, quadoutfn,
                              dosweeps, check, args, argc)) {
        exit(-1);
    }
    return 0;
}
