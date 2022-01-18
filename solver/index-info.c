/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <math.h>

#include "os-features.h"
#include "index.h"
#include "starutil.h"
#include "log.h"
#include "errors.h"
#include "ioutils.h"
#include "boilerplate.h"
#include "tic.h"

static const char* OPTIONS = "hvr:d:R:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <index-files>\n"
           "    [-r <ra>] (deg)\n"
           "    [-d <dec>] (deg)\n"
           "    [-R <radius>] (deg)\n"
           "    [-v]: +verbose\n"
           "\n", progname);
}


int main(int argc, char **argv) {
    int argchar;
    int loglvl = LOG_MSG;
    char** myargs;
    int nmyargs;
    int i;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(argv[0]);
            break;
        default:
            return -1;
        }
    log_init(loglvl);
    nmyargs = argc - optind;
    myargs = argv + optind;

    if (nmyargs < 1) {
        printHelp(argv[0]);
        exit(-1);
    }

    for (i=0; i<nmyargs; i++) {
        char* indexfn = myargs[i];
        index_t index;
        tic();
        logmsg("Reading meta-data for index %s\n", indexfn);
        if (index_get_meta(indexfn, &index)) {
            ERROR("Failed to read metadata for index %s", indexfn);
            continue;
        }
        toc();

        logmsg("Index %s: id %i, healpix %i (nside %i), %i stars, %i quads, dimquads=%i, scales %g to %g arcmin.\n",
               index.indexname,
               index.indexid, index.healpix, index.hpnside,
               index.nstars, index.nquads, index.dimquads,
               arcsec2arcmin(index.index_scale_lower),
               arcsec2arcmin(index.index_scale_upper));

        index_close(&index);
    }

    return 0;
}

