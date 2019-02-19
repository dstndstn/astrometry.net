/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <sys/param.h>
#include <stdio.h>
#include <math.h>

#include "os-features.h"
#include "anwcs.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"
#include "mathutil.h"
#include "keywords.h"
#include "tic.h"
#include "anqfits.h"

static const char* OPTIONS = "hrvz:o:e:w:WqI:x:y:";

void printHelp(char* progname) {
    fprintf(stderr, "%s [options] <input-FITS-image> <output-WCS> <output-FITS-filename>\n"
            "    [-e <ext>]: input FITS extension to read (default: primary extension, 0)\n"
            "    [-w <wcs-file>] (default: input file)\n"
            "    [-o <order>]: Lanczos order (default 3)\n"
            "    [-v]: more verbose\n"
            "    [-q]: less verbose\n"
            "\n", progname);
}


static double lanczos(double x, int order) {
    if (x == 0)
        return 1.0;
    if (x > order || x < -order)
        return 0.0;
    return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
}




static void resample_image(const double* img, int W, int H, anwcs_t* inwcs,
                           double* outimg, int outW, int outH, anwcs_t* outwcs,
                           double* weightimg, double imgweight,
                           anbool set_or_add, int order) {
    //
    double support;

    support = order;

    int i, j;
    for (i=0; i<outH; i++) {
        for (j=0; j<outW; j++) {
            double px, py;
            double weight;
            double sum;
            int x0,x1,y0,y1;
            int ix,iy;
            double ra, dec;

            // +1 for FITS
            if (anwcs_pixelxy2radec(outwcs, j+1, i+1, &ra, &dec)) {
                ERROR("Failed to project pixel (%i,%i) through output WCS\n", j, i);
                continue;
            }
            if (anwcs_radec2pixelxy(inwcs, ra, dec, &px, &py)) {
                ERROR("Failed to project pixel (%i,%i) through input WCS\n", j, i);
                continue;
            }
            // -1 for FITS
            px -= 1;
            py -= 1;

            /* ??
             if (px < -support || px >= W+support)
             continue;
             if (py < -support || py >= H+support)
             continue;
             */
            if (px < 0 || px >= W)
                continue;
            if (py < 0 || py >= H)
                continue;

            x0 = MAX(0, (int)floor(px - support));
            y0 = MAX(0, (int)floor(py - support));
            x1 = MIN(W-1, (int) ceil(px + support));
            y1 = MIN(H-1, (int) ceil(py + support));
            weight = 0.0;
            sum = 0.0;

            for (iy=y0; iy<=y1; iy++) {
                for (ix=x0; ix<=x1; ix++) {
                    double K;
                    double pix;

                    double d;
                    d = hypot(px - ix, py - iy);
                    K = lanczos(d, order);
                    //K = space->kernel(space->token, px, py, ix, iy);
                    if (K == 0)
                        continue;
                    pix = img[iy*W + ix];
                    if (isnan(pix))
                        // out-of-bounds pixel
                        continue;
                    if (!isfinite(pix)) {
                        logverb("Pixel value: %g\n", pix);
                        continue;
                    }
                    weight += K;
                    sum += K * pix;
                }
            }
            if (weight != 0) {
                if (set_or_add)
                    outimg[i*outW + j] = sum / weight;
                else
                    outimg[i*outW + j] += sum / weight;

                if (weightimg)
                    weightimg[i*outW + j] += imgweight;
            }
        }
        logverb("Row %i of %i\n", i+1, outH);
    }
}

int main(int argc, char** args) {
    int argchar;

    char* infn = NULL;
    char* outfn = NULL;
    char* inwcsfn = NULL;
    char* outwcsfn = NULL;

    double* img = NULL;
    double* outimg = NULL;
    int W, H;
    int outW, outH;
    anwcs_t* inwcs;
    anwcs_t* outwcs;

    double inpixscale;
    double outpixscale;
    int i;
    int loglvl = LOG_MSG;
    anqfits_t* anq;
    int fitsext = 0;
    int order = 3;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case '?':
        case 'h':
            printHelp(args[0]);
            exit(0);
        case 'v':
            loglvl++;
            break;
        case 'q':
            loglvl--;
            break;
        case 'e':
            fitsext = atoi(optarg);
            break;
        case 'w':
            inwcsfn = optarg;
            break;
        case 'o':
            order = atoi(optarg);
            break;
        }

    log_init(loglvl);
    fits_use_error_system();

    if (argc - optind != 3) {
        printHelp(args[0]);
        exit(-1);
    }
		
    infn = args[optind];
    outwcsfn = args[optind+1];
    outfn = args[optind+2];
    if (!inwcsfn)
        inwcsfn = infn;

    anq = anqfits_open(infn);
    if (!anq) {
        ERROR("Failed to open \"%s\"", infn);
        exit(-1);
    }

    img = anqfits_readpix(anq, fitsext, 0, 0, 0, 0, 0, PTYPE_DOUBLE,
                          NULL, &W, &H);
    if (!img) {
        ERROR("Failed to read pixel from \"%s\"", infn);
        exit(-1);
    }
    anqfits_close(anq);
    logmsg("Read image %s: %i x %i.\n", infn, W, H);

    logmsg("Reading input WCS file %s\n", inwcsfn);
    inwcs = anwcs_open(inwcsfn, fitsext);
    if (!inwcs) {
        ERROR("Failed to read WCS from file: %s\n", inwcsfn);
        exit(-1);
    }
    inpixscale = anwcs_pixel_scale(inwcs);
    if (inpixscale == 0) {
        ERROR("Pixel scale from the WCS file is zero.  Usually this means the image has no valid WCS header.\n");
        exit(-1);
    }

    logmsg("Reading output WCS file %s\n", outwcsfn);
    outwcs = anwcs_open(outwcsfn, fitsext);
    if (!outwcs) {
        ERROR("Failed to read WCS from file: %s\n", outwcsfn);
        exit(-1);
    }
    outpixscale = anwcs_pixel_scale(outwcs);
    if (inpixscale == 0) {
        ERROR("Pixel scale from the WCS file is zero.  Usually this means the image has no valid WCS header.\n");
        exit(-1);
    }

    logmsg("Input pixel scale: %g.  Output: %g (arcsec/pix)\n", inpixscale, outpixscale);

    outW = anwcs_imagew(outwcs);
    outH = anwcs_imageh(outwcs);

    logmsg("Output image will be %i x %i\n", outW, outH);

    outimg = malloc(outW * outH * sizeof(double));
    for (i=0; i<outW*outH; i++)
        outimg[i] = 1.0 / 0.0;

    resample_image(img, W, H, inwcs,
                   outimg, outW, outH, outwcs,
                   NULL, 0,
                   TRUE, order);
    free(img);

    logmsg("Writing output: %s\n", outfn);
    // HACK -- reduce output image to float, in-place.
    {
        float* fimg = (float*)outimg;
        for (i=0; i<outW*outH; i++)
            fimg[i] = outimg[i];
        if (fits_write_float_image(fimg, outW, outH, outfn)) {
            ERROR("Failed to write output image %s", outfn);
            exit(-1);
        }
    }

    free(outimg);
    return 0;

}


