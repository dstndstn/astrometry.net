/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "wcs-resample.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "bl.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"

const char* OPTIONS = "hw:e:E:x:L:z";

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <input-FITS-image> <output (target) WCS-file> <output-FITS-image>\n"
           "   [-E <input image FITS extension>] (default: 0)\n"
           "   [-w <input WCS file>] (default is to read WCS from input FITS image)\n"
           "   [-e <input WCS FITS extension>] (default: 0)\n"
           "   [-x <output WCS FITS extension>] (default: 0)\n"
           "   [-L <Lanczos order>] (default: nearest-neighbor resampling)\n"
           "   [-z]: zero out inf/nan input image value\n"
           "\n", progname);
}


int main(int argc, char** args) {
    int c;
    char* inwcsfn = NULL;
    char* outwcsfn = NULL;
    char* infitsfn = NULL;
    char* outfitsfn = NULL;
    int inwcsext = 0;
    int inimgext = 0;
    int outwcsext = 0;
    int Lorder = 0;
    int zinf;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'w':
            inwcsfn = optarg;
            break;
        case 'e':
            inwcsext = atoi(optarg);
            break;
        case 'E':
            inimgext = atoi(optarg);
            break;
        case 'x':
            outwcsext = atoi(optarg);
            break;
        case 'L':
            Lorder = atoi(optarg);
            break;
        case 'z':
            zinf = 1;
            break;
        }
    }

    log_init(LOG_MSG);
    fits_use_error_system();

    if (optind != argc - 3) {
        print_help(args[0]);
        exit(-1);
    }

    infitsfn  = args[optind+0];
    outwcsfn  = args[optind+1];
    outfitsfn = args[optind+2];

    if (!inwcsfn)
        inwcsfn = infitsfn;

    if (resample_wcs_files(infitsfn, inimgext, inwcsfn, inwcsext,
                           outwcsfn, outwcsext, outfitsfn, Lorder,
                           zinf)) {
        ERROR("Failed to resample image");
        exit(-1);
    }
    return 0;
}
