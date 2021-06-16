/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "os-features.h"
#include "an-bool.h"
#include "log.h"
#include "errors.h"
#include "mathutil.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "anqfits.h"
#include "qfits_convert.h"
#include "qfits_header.h"

static const char* OPTIONS = "hvs:e:";

static void printHelp(char* progname) {
    printf("%s  [options]  <input-file> <output-file>\n"
           "    use \"-\" to write to stdout.\n"
           "      [-s <scale>]: downsample scale (default: 2): integer\n"
           "      [-e <extension>]: read extension (default: 0)\n"
           "      [-v]: verbose\n"
           "\n", progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    FILE* fout;
    anbool tostdout = FALSE;
    anqfits_t* anq;
    int W, H;
    qfits_header* hdr;
    const anqfits_image_t* animg;
    float* img;
    int loglvl = LOG_MSG;
    int scale = 2;
    int winw;
    int winh;
    int plane;
    int out_bitpix = -32;
    float* outimg;
    int outw, outh;
    int edge = EDGE_TRUNCATE;
    int ext = 0;
    int npixout = 0;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 's':
            scale = atoi(optarg);
            break;
        case 'e':
            ext = atoi(optarg);
            break;
        case '?':
        case 'h':
            printHelp(argv[0]);
            return 0;
        default:
            return -1;
        }

    log_init(loglvl);
    log_to(stderr);
    errors_log_to(stderr);
    fits_use_error_system();

    if (argc - optind != 2) {
        logerr("Need two arguments: input and output files.\n");
        printHelp(argv[0]);
        exit(-1);
    }

    infn = argv[optind];
    outfn = argv[optind+1];

    if (streq(outfn, "-")) {
        tostdout = TRUE;
        fout = stdout;
    } else {
        fout = fopen(outfn, "wb");
        if (!fout) {
            SYSERROR("Failed to open output file \"%s\"", outfn);
            exit(-1);
        }
    }

    anq = anqfits_open(infn);
    if (!anq) {
        ERROR("Failed to open input file \"%s\"", infn);
        exit(-1);
    }
    animg = anqfits_get_image_const(anq, ext);
    
    W = (int)animg->width;
    H = (int)animg->height;
    if (!animg) {
        ERROR("Failde to read image from \"%s\"", infn);
        exit(-1);
    }

    /*
     if (tostdout)
     dump.filename = "STDOUT";
     else
     dump.filename = outfn;
     dump.ptype = PTYPE_FLOAT;
     dump.out_ptype = out_bitpix;
     */

    get_output_image_size(W % scale, H % scale,
                          scale, edge, &outw, &outh);
    outw += (W / scale);
    outh += (H / scale);
    
    hdr = qfits_header_default();
    fits_header_add_int(hdr, "BITPIX", out_bitpix, "bits per pixel");
    if (animg->planes > 1)
        fits_header_add_int(hdr, "NAXIS", 3, "number of axes");
    else
        fits_header_add_int(hdr, "NAXIS", 2, "number of axes");
    fits_header_add_int(hdr, "NAXIS1", outw, "image width");
    fits_header_add_int(hdr, "NAXIS2", outh, "image height");
    if (animg->planes > 1)
        fits_header_add_int(hdr, "NAXIS3", animg->planes, "number of planes");

    if (qfits_header_dump(hdr, fout)) {
        ERROR("Failed to write FITS header to \"%s\"", outfn);
        exit(-1);
    }
    qfits_header_destroy(hdr);

    winw = W;
    winh = (int)ceil(ceil(1024*1024 / (float)winw) / (float)scale) * scale;

    outimg = malloc((size_t)ceil(winw/scale)*(size_t)ceil(winh/scale) * sizeof(float));
			
    logmsg("Image is %i x %i x %i\n", W, H, (int)animg->planes);
    logmsg("Output will be %i x %i x %i\n", outw, outh, (int)animg->planes);
    logverb("Reading in blocks of %i x %i\n", winw, winh);
    for (plane=0; plane<animg->planes; plane++) {
        int bx, by;
        int nx, ny;
        for (by=0; by<(int)ceil(H / (float)winh); by++) {
            for (bx=0; bx<(int)ceil(W / (float)winw); bx++) {
                int i;
                int lox, loy, hix, hiy, outw, outh;
                nx = MIN(winw, W - bx*winw);
                ny = MIN(winh, H - by*winh);
                lox = bx*winw;
                loy = by*winh;
                hix = lox + nx;
                hiy = loy + ny;
                logverb("  reading %i,%i + %i,%i\n", lox, loy, nx, ny);

                img = anqfits_readpix(anq, ext, lox, hix, loy, hiy, plane,
                                      PTYPE_FLOAT, NULL, &W, &H);
                if (!img) {
                    ERROR("Failed to load pixel window: x=[%i, %i), y=[%i,%i), plane %i\n",
                          lox, hix, loy, hiy, plane);
                    exit(-1);
                }

                average_image_f(img, nx, ny, scale, edge,
                                &outw, &outh, outimg);
                free(img);

                logverb("  writing %i x %i\n", outw, outh);
                if (outw * outh == 0)
                    continue;

                for (i=0; i<outw*outh; i++) {
                    int nbytes = abs(out_bitpix)/8;
                    char buf[nbytes];
                    if (qfits_pixel_ctofits(PTYPE_FLOAT, out_bitpix,
                                            outimg + i, buf)) {
                        ERROR("Failed to convert pixel to FITS type\n");
                        exit(-1);
                    }
                    if (fwrite(buf, nbytes, 1, fout) != 1) {
                        ERROR("Failed to write pixels\n");
                        exit(-1);
                    }
                }
                npixout += outw*outh;
            }
        }
    }
    free(outimg);
    anqfits_close(anq);

    if (tostdout) {
        // pad.
        int N;
        char pad[2880];
        N = (npixout * (abs(out_bitpix) / 8)) % 2880;
        memset(pad, 0, 2880);
        fwrite(pad, 1, N, fout);
    } else {
        if (fits_pad_file(fout)) {
            ERROR("Failed to pad output file");
            exit(-1);
        }
        if (fclose(fout)) {
            SYSERROR("Failed to close output file");
            exit(-1);
        }
    }
    logverb("Done!\n");
    return 0;
}
