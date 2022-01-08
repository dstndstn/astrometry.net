/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "os-features.h"
#include "uniformize-catalog.h"
#include "fitstable.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "fitsioutils.h"
#include "mathutil.h"

const char* OPTIONS = "hvH:s:n:N:d:R:D:S:fm:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <input-FITS-catalog> <output-FITS-catalog>\n"
           "    [-R <ra-column-name>]: name of RA in FITS table (default RA)\n"
           "    [-D <dec-column-name>]: name of DEC in FITS table (default DEC)\n"
           "    [-S <sort-column-name>]: column on which to sort\n"
           "    [-f]: sort in descending order (eg, for FLUX); default ascending (eg, for MAG)\n"
           "    [-H <big healpix>]; default is all-sky\n"
           "    [-s <big healpix Nside>]; default is 1\n"
           "    [-m <margin>]: add a margin of <margin> healpixels; default 0\n"
           "    [-n <sweeps>]    (ie, number of stars per fine healpix grid cell); default 10\n"
           "    [-N <nside>]:   fine healpixelization grid; default 100.\n"
           "    [-d <dedup-radius>]: deduplication radius in arcseconds; default no deduplication\n"
           "    [-v]: +verbose\n"
           "\n", progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* progname = argv[0];
    char* infn = NULL;
    char* outfn = NULL;
    char* racol = NULL;
    char* deccol = NULL;
    char* sortcol = NULL;
    anbool sortasc = TRUE;
    int loglvl = LOG_MSG;
    int bighp = -1;
    int bignside = 1;
    int sweeps = 10;
    int Nside = 100;
    double dedup = 0.0;
    int margin = 0;
    double mincut = -LARGE_VAL;
	
    fitstable_t* intable;
    fitstable_t* outtable;

    char** myargs;
    int nmyargs;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'R':
            racol = optarg;
            break;
        case 'D':
            deccol = optarg;
            break;
        case 'S':
            sortcol = optarg;
            break;
        case 'f':
            sortasc = FALSE;
            break;
        case 'H':
            bighp = atoi(optarg);
            break;
        case 's':
            bignside = atoi(optarg);
            break;
        case 'n':
            sweeps = atoi(optarg);
            break;
        case 'N':
            Nside = atoi(optarg);
            break;
        case 'd':
            dedup = atof(optarg);
            break;
        case 'm':
            margin = atoi(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    nmyargs = argc - optind;
    myargs = argv + optind;

    if (nmyargs != 2) {
        printHelp(progname);
        exit(-1);
    }
    log_init(loglvl);
    fits_use_error_system();

    infn = myargs[0];
    outfn = myargs[1];

    logmsg("Reading catalog from %s, writing to %s\n", infn, outfn);

    logmsg("Reading %s...\n", infn);
    intable = fitstable_open(infn);
    if (!intable) {
        ERROR("Couldn't read catalog %s", infn);
        exit(-1);
    }
    logmsg("Got %i stars\n", fitstable_nrows(intable));

    outtable = fitstable_open_for_writing(outfn);
    if (!outtable) {
        ERROR("Failed to open output table %s", outfn);
        exit(-1);
    }
    /*
     if (fitstable_write_primary_header(outtable)) {
     ERROR("Failed to write primary header");
     exit(-1);
     }
     */
    if (uniformize_catalog(intable, outtable, racol, deccol,
                           sortcol, sortasc, mincut,
                           bighp, bignside, margin,
                           Nside, dedup, sweeps,
                           argv, argc)) {
        exit(-1);
    }

    if (fitstable_fix_primary_header(outtable) ||
        fitstable_close(outtable)) {
        ERROR("Failed to close output table");
        exit(-1);
    }

    fitstable_close(intable);
    return 0;
}


