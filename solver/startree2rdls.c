/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "kdtree.h"
#include "starutil.h"
#include "bl.h"
#include "starkd.h"
#include "boilerplate.h"
#include "rdlist.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hr:";


void print_help(char* progname)
{
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "Usage: %s\n"
            "   -r <rdls-output-file>\n"
            "   [-v]: more verbose\n"
            "   [-h]: help\n"
            "   <skdt> [<skdt> ...]\n\n"
            "Reads .skdt files.  Writes an RDLS containing the star locations.\n",
            progname);
}

int main(int argc, char** args) {
    int argchar;
    char* outfn = NULL;
    char* fn;
    rdlist_t* rdls;
    startree_t* skdt = NULL;
    int i;
    int loglvl = LOG_MSG;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 'r':
            outfn = optarg;
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        }

    log_init(loglvl);
    if (!outfn || (optind == argc)) {
        print_help(args[0]);
        exit(-1);
    }

    rdls = rdlist_open_for_writing(outfn);
    if (!rdls) {
        ERROR("Failed to open RDLS file %s for output", outfn);
        exit(-1);
    }
    if (rdlist_write_primary_header(rdls)) {
        ERROR("Failed to write RDLS header");
        exit(-1);
    }

    for (; optind<argc; optind++) {
        int Nstars;
        fn = args[optind];
        logmsg("Opening star kdtree %s...\n", fn);
        skdt = startree_open(fn);
        if (!skdt) {
            ERROR("Failed to read star kdtree %s", fn);
            exit(-1);
        }
        Nstars = startree_N(skdt);

        if (rdlist_write_header(rdls)) {
            ERROR("Failed to write new RDLS field header");
            exit(-1);
        }

        logmsg("Reading stars...\n");
        for (i=0; i<Nstars; i++) {
            double xyz[3];
            double radec[2];
            if (!(i % 200000)) {
                printf(".");
                fflush(stdout);
            }
            startree_get(skdt, i, xyz);
            xyzarr2radecdegarr(xyz, radec);
            if (rdlist_write_one_radec(rdls, radec[0], radec[1])) {
                ERROR("Failed to write a RA,Dec entry");
                exit(-1);
            }
        }
        printf("\n");

        startree_close(skdt);

        if (rdlist_fix_header(rdls)) {
            ERROR("Failed to fix RDLS field header");
            exit(-1);
        }
    }

    if (rdlist_fix_primary_header(rdls) ||
        rdlist_close(rdls)) {
        ERROR("Failed to close RDLS file");
        exit(-1);
    }

    return 0;
}

