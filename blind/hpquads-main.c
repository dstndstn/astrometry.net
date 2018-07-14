/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "hpquads.h"
#include "healpix.h"
#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "tic.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "bt.h"
#include "rdlist.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "quad-utils.h"
#include "quad-builder.h"

static const char* OPTIONS = "hi:c:q:bn:u:l:d:p:r:L:RI:F:HEv";

static void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "      -i <input-filename>    (star kdtree (skdt.fits) input file)\n"
           "      -c <codes-output-filename>    (codes file (code.fits) output file)\n"
           "      -q <quads-output-filename>    (quads file (quad.fits) output file)\n"
           "      -n <nside>     healpix nside\n"
           "      -u <scale>     upper bound of quad scale (arcmin)\n"
           "     [-l <scale>]    lower bound of quad scale (arcmin)\n"
           "     [-d <dimquads>] number of stars in a \"quad\".\n"
           "     [-p <passes>]   number of rounds of quad-building (ie, # quads per healpix cell, default 1)\n"
           "     [-r <reuse-times>] number of times a star can be used.\n"
           "     [-L <max-reuses>] make extra passes through the healpixes, increasing the \"-r\" reuse\n"
           "                     limit each time, up to \"max-reuses\".\n"
           "     [-I <unique-id>] set the unique ID of this index\n\n"
           "     [-E]: scan through the catalog, checking which healpixes are occupied.\n"
           "     [-v]: verbose\n"
           "\nReads skdt, writes {code, quad}.\n\n"
           , progname);
}


int main(int argc, char** argv) {
    int argchar;

    char *quadfn = NULL;
    char *codefn = NULL;
    char *skdtfn = NULL;
    int Nside = 0;
    int id = 0;
    int passes = 1;
    int Nreuse = 3;
    int Nloosen = 0;
    anbool scanoccupied = FALSE;
    int dimquads = 4;
    double scale_min_arcmin = 0.0;
    double scale_max_arcmin = 0.0;
	
    int loglvl = LOG_MSG;

    int i;
	
    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 'E':
            scanoccupied = TRUE;
            break;
        case 'd':
            dimquads = atoi(optarg);
            break;
        case 'L':
            Nloosen = atoi(optarg);
            break;
        case 'r':
            Nreuse = atoi(optarg);
            break;
        case 'p':
            passes = atoi(optarg);
            break;
        case 'I':
            id = atoi(optarg);
            break;
        case 'n':
            Nside = atoi(optarg);
            break;
        case 'i':
            skdtfn = optarg;
            break;
        case 'c':
            codefn = optarg;
            break;
        case 'q':
            quadfn = optarg;
            break;
        case 'u':
            scale_max_arcmin = atof(optarg);
            break;
        case 'l':
            scale_min_arcmin = atof(optarg);
            break;
        case 'h':
            print_help(argv[0]);
            exit(0);
        default:
            return -1;
        }

    log_init(loglvl);

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

    if (!skdtfn || !codefn || !quadfn) {
        printf("Specify in & out filenames, bonehead!\n");
        print_help(argv[0]);
        exit( -1);
    }
    if (dimquads > DQMAX) {
        ERROR("Quad dimension %i exceeds compiled-in max %i.\n", dimquads, DQMAX);
        exit(-1);
    }
    if (scale_max_arcmin == 0.0) {
        ERROR("Must specify maximum quad scale: -u <scale>");
        exit(-1);
    }

    if (!id)
        logmsg("Warning: you should set the unique-id for this index (-i).\n");

    if (hpquads_files(skdtfn, codefn, quadfn, Nside,
                      scale_min_arcmin, scale_max_arcmin,
                      dimquads, passes, Nreuse, Nloosen,
                      id, scanoccupied, 
                      NULL, NULL, 0,
                      argv, argc)) {
        ERROR("hpquads failed");
        exit(-1);
    }
    return 0;
}

	
