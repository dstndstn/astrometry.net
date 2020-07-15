/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

#include "os-features.h"
#include "image2xy-files.h"
#include "log.h"
#include "errors.h"
#include "ioutils.h"

static const char* OPTIONS = "hi:Oo:8Hd:D:ve:B:S:M:s:p:P:bU:g:C:m:a:G:w:L:";

static void printHelp() {
    fprintf(stderr,
            "Usage: image2xy [options] fitsname.fits \n"
            "\n"
            "Read a FITS file, find objects, and write out \n"
            "X, Y, FLUX to   fitsname.xy.fits .\n"
            "\n"
            "   [-e <extension>]: read from a single FITS extension\n"
            "   [-O]  overwrite existing output file.\n"
            "   [-o <output-filename>]  write XYlist to given filename.\n"
            "   [-L <Lanczos-order>]\n"
            "   [-8]  don't use optimization for byte (u8) images.\n"
            "   [-H]  downsample by a factor of 2 before running simplexy.\n"
            "   [-d <downsample-factor>]  downsample by an integer factor before running simplexy.\n"
            "   [-D <downsample-factor>] downsample, if necessary, by this many factors of two.\n"
            "   [-s <median-filtering scale>]: set median-filter box size (default %i pixels)\n"
            "   [-w <PSF width>]: set Gaussian PSF sigma (default %g pixel)\n"
            "   [-g <sigma>]: set image noise level\n"
            "   [-p <sigmas>]: set significance level of peaks (default %g sigmas)\n"
            "   [-a <saddle-sigmas>]: set \"saddle\" level joining peaks (default %g sigmas)\n"
            "   [-P <image plane>]: pull out a single plane of a multi-color image (default: first plane)\n"
            "   [-b]: don't do (median-based) background subtraction\n"
            "   [-G <background>]: subtract this 'global' background value; implies -b\n"
            "   [-m]: set maximum extended object size for deblending (default %i pixels)\n"
            "\n"
            "   [-S <background-subtracted image>]: save background-subtracted image to this filename (FITS float image)\n"
            "   [-B <background image>]: save background image to filename\n"
            "   [-U <smoothed background-subtracted image>]: save smoothed background-subtracted image to filename\n"
            "   [-M <mask image>]: save mask image to filename\n"
            "   [-C <blob-image>]: save connected-components image to filename\n"
            "\n"
            "   [-v] verbose - repeat for more and more verboseness\n"
            "\n"
            "   image2xy 'file.fits[1]'   - process first extension.\n"
            "   image2xy 'file.fits[2]'   - process second extension \n"
            "   image2xy file.fits+2      - same as above \n"
            "\n",
            SIMPLEXY_DEFAULT_HALFBOX,
            SIMPLEXY_DEFAULT_DPSF,
            SIMPLEXY_DEFAULT_PLIM,
            SIMPLEXY_DEFAULT_SADDLE,
            SIMPLEXY_DEFAULT_MAXSIZE);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* outfn = NULL;
    char* infn;
    int overwrite = 0;
    int loglvl = LOG_MSG;
    anbool do_u8 = TRUE;
    int downsample = 0;
    int downsample_as_reqd = 0;
    int extension = 0;
    int plane = 0;

    simplexy_t sparams;
    simplexy_t* params = &sparams;

    memset(params, 0, sizeof(simplexy_t));

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1) {
        switch (argchar) {
        case 'L':
            params->Lorder = atoi(optarg);
            break;
        case 'w':
            params->dpsf = atof(optarg);
            break;
        case 'a':
            params->saddle = atof(optarg);
            break;
        case 'm':
            params->maxsize = atoi(optarg);
            break;
        case 'g':
            params->sigma = atof(optarg);
            break;
        case 'b':
            params->nobgsub = TRUE;
            break;
        case 'G':
            params->nobgsub = TRUE;
            params->globalbg = atof(optarg);
            break;
        case 'P':
            plane = atoi(optarg);
            break;
        case 's':
            params->halfbox = atoi(optarg);
            break;
        case 'p':
            params->plim = atof(optarg);
            break;
        case 'B':
            params->bgimgfn = optarg;
            break;
        case 'S':
            params->bgsubimgfn = optarg;
            break;
        case 'M':
            params->maskimgfn = optarg;
            break;
        case 'U':
            params->smoothimgfn = optarg;
            break;
        case 'C':
            params->blobimgfn = optarg;
            break;
        case 'e':
            extension = atoi(optarg);
            break;
        case 'D':
            downsample_as_reqd = atoi(optarg);
            break;
        case 'H':
            downsample = 2;
            break;
        case 'd':
            downsample = atoi(optarg);
            break;
        case '8':
            do_u8 = FALSE;
            break;
        case 'v':
            loglvl++;
            break;
        case 'O':
            overwrite = 1;
            break;
        case 'o':
            outfn = strdup(optarg);
            break;
        case '?':
        case 'h':
            printHelp();
            exit(0);
        }
    }
    
    if (optind != argc - 1) {
        printHelp();
        exit(-1);
    }

    infn = argv[optind];

    log_init(loglvl);
    logverb("infile=%s\n", infn);

    if (!outfn) {
        // Create xylist filename (by trimming '.fits')
        asprintf_safe(&outfn, "%.*s.xy.fits", (int)(strlen(infn)-5), infn);
        logverb("outfile=%s\n", outfn);
    }

    if (overwrite && file_exists(outfn)) {
        logverb("Deleting existing output file \"%s\"...\n", outfn);
        if (unlink(outfn)) {
            SYSERROR("Failed to delete existing output file \"%s\"", outfn);
            exit(-1);
        }
    }

    if (downsample)
        logverb("Downsampling by %i\n", downsample);

    if (image2xy_files(infn, outfn, do_u8, downsample, downsample_as_reqd,
                       extension, plane, params)) {
        ERROR("image2xy failed.");
        exit(-1);
    }
    free(outfn);
    return 0;
}
