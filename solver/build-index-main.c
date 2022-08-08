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

#include "build-index.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "starutil.h"

const char* OPTIONS = "hvi:o:N:l:u:S:fU:H:s:m:n:r:d:p:R:L:EI:MTj:1:P:B:A:D:Kt:e:";

static void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "      (\n"
           "         -i <input-FITS-catalog>  input: source RA,DEC, etc\n"
           "         -e <input-FITS-extension> input: FITS extension to read\n"
           "    OR,\n"
           "         -1 <input-index>         to share another index's stars\n"
           "      )\n"
           "      -o <output-index>        output filename for index\n"
           "      (\n"
           "         -P <scale-number>: use 'preset' values for '-N', '-l', and '-u'\n"
           "               (the scale-number is the last two digits of the pre-cooked\n"
           "                index filename -- eg, index-205 is  \"-P 5\".\n"
           "                -P 0  should be good for images about 6 arcmin in size\n"
           "                    and it goes in steps of sqrt(2), so:\n"
           "                -P 2  should work for images about 12 arcmin across\n"
           "                -P 4  should work for images about 24 arcmin across\n"
           "                -P 6  should work for images about 1 degree across\n"
           "                -P 8  should work for images about 2 degree across\n"
           "                -P 10 should work for images about 4 degree across\n"
           "                 etc... up to -P 19\n"
           "  OR,\n"
           "         -N <nside>            healpix Nside for quad-building (-P 0 -> -N 1760)\n"
           "         -l <min-quad-size>    minimum quad size (arcminutes) (-P 0 -> -l 2.0)\n"
           "         -u <max-quad-size>    maximum quad size (arcminutes) (-P 0 -> -u 2.8)\n"
           "      )\n"
           "      [-S <column>]: sort column (default: assume the input file is already sorted)\n"
           "      [-f]: sort in descending order (eg, for FLUX); default ascending (eg, for MAG)\n"
           "      [-A <column>]: specify the RA  column name in the input FITS table (default \"RA\")\n"
           "      [-D <column>]: specify the Dec column name in the input FITS table (default \"Dec\")\n"
           "      [-B <val>]: cut any object whose sort-column value is less than 'val'; for mags this is a bright limit\n"
           "      [-K]: keep original RA,Dec columns in the tag-along table.\n"
           "      [-U]: healpix Nside for uniformization (default: same as -N)\n"
           "      [-H <big healpix>]; default is all-sky\n"
           "      [-s <big healpix Nside>]; default is 1\n"
           "      [-m <margin>]: add a margin of <margin> healpixels; default 0\n"
           "      [-n <sweeps>]    (ie, number of stars per fine healpix grid cell); default 10\n"
           "      [-r <dedup-radius>]: deduplication radius in arcseconds; default no deduplication\n"
           "      [-j <jitter-arcsec>]: positional error of stars in the reference catalog (in arcsec; default 1)\n"
           "\n"
           "      [-d <dimquads>] number of stars in a \"quad\" (default 4).\n"
           "      [-p <passes>]   number of rounds of quad-building (ie, # quads per healpix cell, default 16)\n"
           "      [-R <reuse-times>] number of times a star can be used (default: 8)\n"
           "      [-L <max-reuses>] make extra passes through the healpixes, increasing the \"-r\" reuse\n"
           "                     limit each time, up to \"max-reuses\".\n"
           "      [-E]: scan through the catalog, checking which healpixes are occupied.\n"
           "\n"
           "      [-I <unique-id>] set the unique ID of this index\n"
           "\n"
           "      [-M]: in-memory (don't use temp files)\n"
           "      [-T]: don't delete temp files\n"
           "      [-t <temp-dir>]: use this temp directory (default: /tmp)\n"
           "      [-v]: add verbosity.\n"
           "\n", progname);
}


int main(int argc, char** argv) {
    int argchar;

    char* infn = NULL;
    char* indexfn = NULL;
    char* inindexfn = NULL;
    int inext = 0;

    index_params_t myp;
    index_params_t* p;

    int loglvl = LOG_MSG;
    int i;
    int preset = -100;

    p = &myp;
    build_index_defaults(p);

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'A':
            p->racol = optarg;
            break;
        case 'D':
            p->deccol = optarg;
            break;
        case 'B':
            p->brightcut = atof(optarg);
            break;
        case 'K':
            p->drop_radec = FALSE;
            break;
        case 'P':
            preset = atoi(optarg);
            break;
        case '1':
            inindexfn = optarg;
            break;
        case 'j':
            p->jitter = atof(optarg);
            break;
        case 't':
            p->tempdir = optarg;
            break;
        case 'T':
            p->delete_tempfiles = FALSE;
            break;
        case 'E':
            p->scanoccupied = TRUE;
            break;
        case 'L':
            p->Nloosen = atoi(optarg);
            break;
        case 'R':
            p->Nreuse = atoi(optarg);
            break;
        case 'p':
            p->passes = atoi(optarg);
            break;
        case 'M':
            p->inmemory = TRUE;
            break;
        case 'U':
            p->UNside = atoi(optarg);
            break;
        case 'i':
            infn = optarg;
            break;
        case 'e':
            inext = atoi(optarg);
            break;
        case 'o':
            indexfn = optarg;
            break;
        case 'u':
            p->qhi = atof(optarg);
            break;
        case 'l':
            p->qlo = atof(optarg);
            break;
        case 'S':
            p->sortcol = optarg;
            break;
        case 'f':
            p->sortasc = FALSE;
            break;
        case 'H':
            p->bighp = atoi(optarg);
            break;
        case 's':
            p->bignside = atoi(optarg);
            break;
        case 'n':
            p->sweeps = atoi(optarg);
            break;
        case 'N':
            p->Nside = atoi(optarg);
            break;
        case 'r':
            p->dedup = atof(optarg);
            break;
        case 'm':
            p->margin = atoi(optarg);
            break;
        case 'd':
            p->dimquads = atoi(optarg);
            break;
        case 'I':
            p->indexid = atoi(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case 'h':
            print_help(argv[0]);
            exit(0);
        default:
            return -1;
        }
	
    log_init(loglvl);

    if (!(infn || inindexfn) || !indexfn) {
        printf("You must specify input & output filenames.\n");
        print_help(argv[0]);
        exit( -1);
    }
    if (infn && inindexfn) {
        printf("Only specify one of -i <input catalog> and -1 <share star kdtree>!\n");
        print_help(argv[0]);
        exit( -1);
    }

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

    if (!p->indexid)
        logmsg("Warning: you should set the unique-id for this index (with -I).\n");

    if (p->dimquads > DQMAX) {
        ERROR("Quad dimension %i exceeds compiled-in max %i.\n", p->dimquads, DQMAX);
        exit(-1);
    }

    if (preset > -100) {
        // presets of -6 and below require 64-bit healpixes
        const int minpreset = -10;
        double scales[] = {
            0.06, 0.09, 0.13, 0.18, 0.25,
            0.35, 0.5, 0.7, 1., 1.4,
            2., 2.8, 4., 5.6, 8., 11., 16., 22., 30., 42., 60., 85.,
            120., 170., 240., 340., 480., 680., 1000., 1400., 2000. };
        // don't need to change this when "minpreset" changes
        double hpbase = 1760;
        double nside;
        int P = sizeof(scales)/sizeof(double) - 1;
        int maxpreset = P + minpreset;
        int prei = preset - minpreset;

        if (preset >= maxpreset) {
            ERROR("Error: only presets %i through %i are defined.\n", minpreset, maxpreset-1);
            exit(-1);
        }
        if (preset < minpreset) {
            ERROR("Preset must be >= %i\n", minpreset);
            exit(-1);
        }

        p->qlo = scales[prei];
        p->qhi = scales[prei+1];
        nside = hpbase * pow((1./sqrt(2)), preset);
        logverb("nside: %g\n", nside);
        if (p->bignside)
            p->Nside = (int)(p->bignside * ceil(nside / (double)p->bignside));
        else
            p->Nside = (int)ceil(nside);
        logverb("Preset %i: quad scales %g to %g, Nside %i\n", preset, p->qlo, p->qhi, p->Nside);
    }

    // For HISTORY cards in output...
    p->argc = argc;
    p->args = argv;

    if (infn) {
        if (build_index_files(infn, inext, indexfn, p)) {
            exit(-1);
        }
    } else {
        if (build_index_shared_skdt_files(inindexfn, indexfn, p)) {
            exit(-1);
        }
    }
    return 0;
}

