/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

/**
   Reads a FITS file, tries to interpret a WCS header and writes out a TAN
   interpretation of it.
*/

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include "qfits.h"
#include "sip.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "starutil.h"

static char* OPTIONS = "ho:";

static void printHelp(char* progname) {
    printf("%s <input-file>\n"
           "   [-o <output-file>]\n"
           "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    char* progname = argv[0];
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    qfits_header* hdr;
    tan_t wcs;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        case 'o':
            outfn = optarg;
            break;
        default:
            return -1;
        }

    if (optind != (argc - 1)) {
        printHelp(progname);
        exit(-1);
    }
    infn = argv[optind];

    hdr = qfits_header_read(infn);
    if (!hdr) {
        fprintf(stderr, "Failed to read FITS header.\n");
        exit(-1);
    }

    memset(&wcs, 0, sizeof(tan_t));

    // This sucks :)

    if (!tan_read_header(hdr, &wcs)) {
        // Couldn't get a wcs from this file.
        return 0;
    }
    qfits_header_destroy(hdr);

    printf("crval1 %g\n", wcs.crval[0]);
    printf("crval2 %g\n", wcs.crval[1]);
    printf("crpix1 %g\n", wcs.crpix[0]);
    printf("crpix2 %g\n", wcs.crpix[1]);
    printf("cd11 %g\n", wcs.cd[0][0]);
    printf("cd12 %g\n", wcs.cd[0][1]);
    printf("cd21 %g\n", wcs.cd[1][0]);
    printf("cd22 %g\n", wcs.cd[1][1]);

    if (outfn) {
        FILE* fout;
        bool tostdout;
        tostdout =  !strcmp(outfn, "-");
        if (tostdout)
            fout = stdout;
        else {
            fout = fopen(outfn, "wb");
            if (!fout) {
                fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
                exit(-1);
            }
        }
        hdr = tan_create_header(&wcs);
        if (!hdr) {
            fprintf(stderr, "Failed to create WCS header.\n");
            exit(-1);
        }
        if (qfits_header_dump(hdr, fout)) {
            fprintf(stderr, "Failed to write WCS header.\n");
            exit(-1);
        }
        if (!tostdout) {
            if (fclose(fout)) {
                fprintf(stderr, "Failed to close output file %s: %s\n", outfn, strerror(errno));
                exit(-1);
            }
        }
    }

    return 0;
}
