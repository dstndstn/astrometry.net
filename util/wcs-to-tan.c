/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include "os-features.h"
#include "starutil.h"
#include "mathutil.h"
#include "fit-wcs.h"
#include "xylist.h"
#include "rdlist.h"
#include "boilerplate.h"
#include "sip.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "anwcs.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hw:e:tLx:y:W:H:N:o:v";

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -w <WCS input file>\n"
           "     [-e <extension>] FITS HDU number to read WCS from (default 0 = primary)\n"
           "     [-t]: just use TAN projection, even if SIP extension exists.\n"
           "     [-L]: force WCSlib\n"
           "   [-x x-lo]\n"
           "   [-y y-lo]\n"
           "   [-W x-hi]\n"
           "   [-H y-hi]\n"
           "   [-N grid-n]\n"
           "   -o <WCS output file>\n"
           "   [-v]: verbose\n"
           "\n", progname);
}


int main(int argc, char** args) {
    int c;
    char* wcsfn = NULL;
    char* outfn = NULL;
    int ext = 0;
    anbool forcetan = FALSE;
    anbool forcewcslib = FALSE;
    double xlo = 1;
    double ylo = 1;
    double xhi = LARGE_VAL;
    double yhi = LARGE_VAL;
    int N = 20;

    double* xyz = NULL;
    double* xy = NULL;

    tan_t outwcs;
    anwcs_t* inwcs = NULL;
    int i, j;
    double xstep, ystep;
    int loglvl = LOG_MSG;

    fits_use_error_system();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'w':
            wcsfn = optarg;
            break;
        case 'L':
            forcewcslib = TRUE;
            break;
        case 't':
            forcetan = TRUE;
            break;
        case 'o':
            outfn = optarg;
            break;
        case 'e':
            ext = atoi(optarg);
            break;
        case 'N':
            N = atoi(optarg);
            break;
        case 'x':
            xlo = atof(optarg);
            break;
        case 'y':
            ylo = atof(optarg);
            break;
        case 'W':
            xhi = atof(optarg);
            break;
        case 'H':
            yhi = atof(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        }
    }
    if (optind != argc) {
        print_help(args[0]);
        exit(-1);
    }
    if (!wcsfn || !outfn) {
        print_help(args[0]);
        exit(-1);
    }
    log_init(loglvl);

    // read WCS.
    logmsg("Trying to read WCS header from \"%s\" ext %i...\n", wcsfn, ext);
    if (forcewcslib) {
        inwcs = anwcs_open_wcslib(wcsfn, ext);
    } else if (forcetan) {
        inwcs = anwcs_open_tan(wcsfn, ext);
    } else {
        inwcs = anwcs_open(wcsfn, ext);
    }
    if (!inwcs) {
        ERROR("Failed to read WCS file \"%s\", extension %i", wcsfn, ext);
        exit(-1);
    }
    logverb("Read WCS:\n");
    if (log_get_level() >= LOG_VERB) {
        anwcs_print(inwcs, log_get_fid());
    }

    if (xhi == LARGE_VAL) {
        xhi = anwcs_imagew(inwcs);
        logverb("Setting image width to %g\n", xhi);
    }
    if (yhi == LARGE_VAL) {
        yhi = anwcs_imageh(inwcs);
        logverb("Setting image height to %g\n", yhi);
    }
    // FIXME -- what if the user wants xhi or yhi == 0?
    if (xhi == LARGE_VAL || xhi == 0) {
        ERROR("Couldn't find the image size; please supply -W\n");
        exit(-1);
    }
    if (yhi == LARGE_VAL || yhi == 0) {
        ERROR("Couldn't find the image size; please supply -H\n");
        exit(-1);
    }

    xstep = (xhi - xlo) / (N-1);
    ystep = (yhi - ylo) / (N-1);

    logverb("Evaluating WCS on a grid of %i x %i in X [%g,%g], Y [%g,%g]\n",
            N, N, xlo, xhi, ylo, yhi);

    xyz = (double*)malloc(sizeof(double) * 3 * N*N);
    xy = (double*)malloc(sizeof(double) * 2 * N*N);
    if (!xyz || !xy) {
        ERROR("Failed to allocate %i xyz, xy coords", N*N);
        exit(-1);
    }

    for (j=0; j<N; j++) {
        for (i=0; i<N; i++) {
            double x, y;
            x = xlo + xstep * i;
            y = ylo + ystep * j;
            if (anwcs_pixelxy2xyz(inwcs, x, y, xyz + (j*N + i)*3)) {
                ERROR("Failed to apply WCS to pixel coord (%g,%g)", x, y);
                exit(-1);
            }
            xy[(j*N+i)*2 + 0] = x;
            xy[(j*N+i)*2 + 1] = y;
        }
    }

    logverb("Fitting TAN WCS\n");
    if (fit_tan_wcs(xyz, xy, N*N, &outwcs, NULL)) {
        ERROR("Failed to fit TAN WCS");
        exit(-1);
    }

    if (tan_write_to_file(&outwcs, outfn)) {
        ERROR("Failed to write TAN WCS header to file \"%s\"", outfn);
        exit(-1);
    }

    free(xy);
    free(xyz);
    anwcs_free(inwcs);

    return 0;
}
