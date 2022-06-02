/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

// Convert UCAC5 z??? files to FITS
// Author: Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>

#include <bzlib.h>

#include "ucac5-fits.h"
#include "ucac5.h"
#include "anqfits.h"
#include "healpix.h"
#include "healpix-utils.h"
#include "bl.h"
#include "starutil.h"
#include "fitsioutils.h"
#include "log.h"
#include "errors.h"
#include "boilerplate.h"

#define OPTIONS "ho:N:e:m:f:"

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage:\n"
           "  %s -o <output-filename-template>  (default: ucac5_%%02i.fits)\n"
           "  [-N <healpix-nside>]              (default = 0)\n"
           "  [-e <epoch in years>]             (default = UCAC epoch)\n"
           "  [-m <margin in degrees>]          (default = 0)\n"
           "  [-f <1 = include tag-along data>] (default = 0)\n"
           "  <input-file> [<input-file> ...]\n"
           "\n"
           "The output-filename-template should contain a \"printf\" sequence like \"%%03i\";\n"
           "we use \"sprintf(filename, output-filename-template, healpix)\"\n"
           "to determine the filename to be used for each healpix.\n\n"
           "-N 0 means output the whole catalog to a single all-sky healpix, which is useful\n"
           "for large-scale indexes; in this case, margin is ignored.\n\n"
           "\nNOTE: WE ASSUME THE UCAC5 FILES ARE GIVEN ON THE COMMAND LINE IN ORDER:\n"
           "z001, z002, ..., z900.\n\n",
           progname);
}


int main(int argc, char** args) {
    char* outfn = "ucac5_%02i.fits";
    int c;
    int startoptind;
    int nrecords, nfiles;
    int Nside = 0;
    float epoch = 0.0;
    double margin = 0.0;
    anbool full = FALSE;

    ucac5_fits** ucacs;

    int i, HP;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
            case '?':
            case 'h':
                print_help(args[0]);
                exit(0);
            case 'N':
                Nside = atoi(optarg);
                break;
            case 'e':
                epoch = atof(optarg);
                break;
            case 'm':
                margin = atof(optarg);
                break;
            case 'f':
                full = atoi(optarg);
                break;
            case 'o':
                outfn = optarg;
                break;
        }
    }

    log_init(LOG_MSG);
    if (!outfn || (optind == argc)) {
        print_help(args[0]);
        exit(-1);
    }

    if (Nside < 0) {
        fprintf(stderr, "Nside must be >= 0.\n");
        print_help(args[0]);
        exit(-1);
    }
    if (Nside) {
        HP = 12 * Nside * Nside;
        printf("Nside = %i, using %i healpixes.\n", Nside, HP);
    }
    else {
        HP = 1;
        printf("Using one all-sky healpix.\n");
    }
    ucacs = calloc(HP, sizeof(ucac5_fits*));
    nrecords = 0;
    nfiles = 0;

    startoptind = optind;
    for (; optind<argc; optind++) {
        char* infn;
        FILE* fid;
        int i;

        infn = args[optind];
        printf("Reading %s\n", infn);
        if ((optind > startoptind) && ((optind - startoptind) % 100 == 0)) {
            printf("\nReading file %i of %i: %s\n", optind - startoptind,
                   argc - startoptind, infn);
        }
        fflush(stdout);

        fid = fopen(infn, "rb");
        if (!fid) {
            SYSERROR("Couldn't open input file \"%s\"", infn);
            exit(-1);
        }

        for (i=0;; i++) {
            ucac5_entry entry;
            il *hplist;
            char buf[UCAC5_RECORD_SIZE];
            int nr;
            anbool eof = 0;

            nr = fread(buf, UCAC5_RECORD_SIZE, 1, fid);
            if (!nr) {
                if (feof(fid))
                    eof = TRUE;
                else {
                    SYSERROR("Error reading input file \"%s\".", infn);
                    exit(-1);
                }
            }

            if (ucac5_parse_entry(&entry, buf, epoch)) {
                ERROR("Failed to parse UCAC5 entry %i in file \"%s\".", i, infn);
                exit(-1);
            }

            if (Nside) {
                if (margin > 0.0)
                    hplist = healpix_rangesearch_radec(entry.ra, entry.dec, margin, Nside, NULL);
                else {
                    int hp = radecdegtohealpix(entry.ra, entry.dec, Nside);
                    hplist = il_new(1);
                    il_append(hplist, hp);
                }
            }
            else {
                hplist = il_new(1);
                il_append(hplist, 0);
            }
            int ihp;
            for (ihp=0; ihp<il_size(hplist); ihp++) {
                int hp = il_get(hplist, ihp);
                if (!ucacs[hp]) {
                    char fn[256];
                    sprintf(fn, outfn, hp);
                    ucacs[hp] = ucac5_fits_open_for_writing(fn, full);
                    if (!ucacs[hp]) {
                        ERROR("Failed to initialize FITS file %i (filename %s)", hp, fn);
                        exit(-1);
                    }
                    fits_header_add_int(ucacs[hp]->header, "HEALPIX", hp, "The healpix number of this catalog.");
                    fits_header_add_int(ucacs[hp]->header, "NSIDE", Nside ? Nside : 1, "The healpix resolution.");
                    BOILERPLATE_ADD_FITS_HEADERS(ucacs[hp]->header);
                    qfits_header_add(ucacs[hp]->header, "HISTORY", "Created by the program \"ucac5tofits\"", NULL, NULL);
                    qfits_header_add(ucacs[hp]->header, "HISTORY", "ucac5tofits command line:", NULL, NULL);
                    fits_add_args(ucacs[hp]->header, args, argc);
                    qfits_header_add(ucacs[hp]->header, "HISTORY", "(end of command line)", NULL, NULL);
                    if (ucac5_fits_write_headers(ucacs[hp])) {
                        ERROR("Failed to write header for FITS file %s", fn);
                        exit(-1);
                    }
                }
                if (ucac5_fits_write_entry(ucacs[hp], &entry)) {
                    ERROR("Failed to write FITS entry");
                    exit(-1);
                }
                nrecords++;
            }
            il_free(hplist);

            if (eof)
                break;
        }

        fclose(fid);

        nfiles++;
        printf("\n");
    }
    printf("\n");

    // close all the files...
    for (i=0; i<HP; i++) {
        if (!ucacs[i])
            continue;
        if (ucac5_fits_fix_headers(ucacs[i]) ||
            ucac5_fits_close(ucacs[i])) {
            ERROR("Failed to close file %i", i);
        }
    }
    printf("Read %u files, %u records.\n", nfiles, nrecords);
    free(ucacs);
    return 0;
}
