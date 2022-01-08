/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "os-features.h"
#include "an-bool.h"
#include "anqfits.h"
#include "permutedsort.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"
#include "mathutil.h"

static const char* OPTIONS = "hi:o:Oe:p:m:IX:N:xnrsvML:H:";

static void printHelp(char* progname) {
    printf("%s    -i <input-file>\n"
           "      [-o <output-file>]       (default stdout)\n"
           "      [-e <extension-number>]  FITS extension (default 0)\n"
           "      [-p <plane-number>]      Image plane number (default 0)\n"
           "      [-m <margin>]            Number of pixels to avoid at the image edges (default 0)\n"
           "      [-O]: do ordinal transform (default: map 25-95 percentile)\n"
           "      [-L <low-percentile>]: set percentile that becomes black (default 25)\n"
           "      [-H <high-percentile>]: set percentile that becomes white (default 95)\n"
           "      [-I]: invert black-on-white image\n"
           "      [-X <max>]: set the input value that will become white\n"
           "      [-N <min>]: set the input value that will become black\n"
           "      [-x]: set max to the observed maximum value\n"
           "      [-n]: set min to the observed minimum value\n"
           "      [-r]: same as -x -n: set min and max to observed data range.\n"
           "      [-s]: write 16-bit output\n"
           "      [-v]: verbose\n"
           "      [-M]: compute & print median value\n"
           "\n", progname);
}


static void sample_percentiles(const float* img, int nx, int ny, int margin,
                               int NPIX, float lop, float hip,
                               float* lo, float* hi) {
    // the maximum number of pixels to sample
    int n, np;
    int x, y;
    int i;
    float* pix;

    //fprintf(stderr, "Computing image percentiles...\n");
    n = (nx - 2*margin) * (ny - 2*margin);
    np = MIN(n, NPIX);
    pix = malloc(np * sizeof(float));
    if (n < NPIX) {
        i=0;
        for (y=margin; y<(ny-margin); y++)
            for (x=margin; x<(nx-margin); x++) {
                pix[i] = img[y*nx + x];
                i++;
            }
        assert(i == np);
    } else {
        for (i=0; i<np; i++) {
            x = margin + (nx - 2*margin) * ( (double)random() / (((double)RAND_MAX)+1.0) );
            y = margin + (ny - 2*margin) * ( (double)random() / (((double)RAND_MAX)+1.0) );
            // On Solaris, apparently, random() / RAND_MAX can be a
            // huge number.  This should have no effect on other
            // machines.
            x = x % nx;
            y = y % ny;

            pix[i] = img[y*nx + x];
        }
    }

    QSORT_R(pix, np, sizeof(float), NULL, compare_floats_asc_r);

    if (lo) {
        i = MIN(np-1, MAX(0, (int)(lop * np)));
        *lo = pix[i];
    }
    if (hi) {
        i = MIN(np-1, MAX(0, (int)(hip * np)));
        *hi = pix[i];
    }
    free(pix);
}
							   



#define NBUF 1024

int main(int argc, char *argv[]) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    FILE* fout = NULL;
    char* progname = argv[0];
    anbool ordinal = FALSE;
    int ext = 0;
    int plane = 0;
    int margin = 0;
    float* img;
    int nx, ny;
    anbool invert = FALSE;

    anqfits_t* anq;

    anbool minval_set = FALSE;
    anbool maxval_set = FALSE;
    float maxval, minval;
    anbool find_min = FALSE;
    anbool find_max = FALSE;

    anbool sixteenbit = FALSE;
    int maxpix;
    int loglvl = LOG_MSG;
    anbool median = FALSE;
    double lop = 0.25;
    double hip = 0.95;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
	case 'L':
            lop = 0.01 * atof(optarg);
            break;
	case 'H':
            hip = 0.01 * atof(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case 's':
            sixteenbit = TRUE;
            break;
        case 'X':
            maxval = atof(optarg);
            maxval_set = TRUE;
            break;
        case 'N':
            minval = atof(optarg);
            minval_set = TRUE;
            break;
        case 'x':
            find_max = TRUE;
            break;
        case 'n':
            find_min = TRUE;
            break;
        case 'r':
            find_min = TRUE;
            find_max = TRUE;
            break;
        case 'I':
            invert = TRUE;
            break;
        case 'i':
            infn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case 'O':
            ordinal = TRUE;
            break;
        case 'e':
            ext = atoi(optarg);
            break;
        case 'p':
            plane = atoi(optarg);
            break;
        case 'm':
            margin = atoi(optarg);
            break;
        case 'M':
            median = TRUE;
            break;
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (!infn) {
        printHelp(progname);
        exit(-1);
    }

    log_init(loglvl);
    log_to(stderr);
    errors_log_to(stderr);
    fits_use_error_system();

    if (outfn) {
        fout = fopen(outfn, "wb");
        if (!fout) {
            SYSERROR("Failed to open output file \"%s\"", outfn);
            exit(-1);
        }
    } else {
        fout = stdout;
    }

    maxpix = (sixteenbit ? 65535 : 255);

    anq = anqfits_open(infn);
    if (!anq) {
        ERROR("Failed to read input file: \"%s\"", infn);
        exit(-1);
    }
    logverb("Reading pixels...\n");
    img = anqfits_readpix(anq, ext, 0,0,0,0, plane,
                          PTYPE_FLOAT, NULL, &nx, &ny);
    if (!img) {
        ERROR("Failed to load pixels.");
        exit(-1);
    }

    if (median) {
        int* perm = permuted_sort(img, sizeof(float), compare_floats_asc, NULL, nx*ny);
        logmsg("Median value: %g\n", img[perm[(nx*ny)/2]]);
        free(perm);
    }

    if (ordinal) {
        int* perm;
        unsigned char* outimg;
        int i;
        int np = nx*ny;

        logverb("Doing ordinal transform...\n");
        perm = permuted_sort(img, sizeof(float), compare_floats_asc, NULL, np);

        if (sixteenbit)
            outimg = malloc(np * sizeof(uint16_t));
        else
            outimg = malloc(np);

        if (invert) {
            for (i=0; i<np; i++)
                outimg[perm[i]] = (unsigned char)(maxpix * (double)(np-1 - i) / (double)(np));
        } else {
            for (i=0; i<np; i++)
                outimg[perm[i]] = (unsigned char)(maxpix * (double)i / (double)(np));
        }
        free(perm);

        logverb("Writing output...\n");
        fprintf(fout, "P5 %i %i %i\n", nx, ny, maxpix);
        if (sixteenbit)
            for (i=0; i<np; i++)
                outimg[i] = htons(outimg[i]);

        if (fwrite(outimg, sixteenbit ? 2 : 1, np, fout) != np) {
            fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
            exit(-1);
        }
        free(outimg);

    } else {
        int i, j;
        float scale;

        if (find_min) {
            minval = LARGE_VALF;
            for (i=0; i<(nx*ny); i++) {
                if (isfinite(img[i])) {
                    minval = MIN(minval, img[i]);
                }
            }
            minval_set = TRUE;
            logverb("Minimum pixel value: %g\n", minval);
        }
        if (find_max) {
            maxval = -LARGE_VALF;
            for (i=0; i<(nx*ny); i++) {
                if (isfinite(img[i])) {
                    maxval = MAX(maxval, img[i]);
                }
            }
            maxval_set = TRUE;
            logverb("Maximum pixel value: %g\n", maxval);
        }

        if (!(minval_set && maxval_set)) {
            int NPIX = 10000;

            // percentiles.
            if (invert) {
                double tmp = 1 - hip;
                hip = 1 - lop;
                lop = tmp;
            }

            logverb("Computing image percentiles...\n");
            sample_percentiles(img, nx, ny, margin, NPIX, lop, hip,
                               (minval_set ? NULL : &minval),
                               (maxval_set ? NULL : &maxval));
        }

        if (invert) {
            scale = -((float)maxpix / (maxval - minval));
            minval = maxval;
        } else
            scale = ((float)maxpix / (maxval - minval));

        logverb("Mapping input pixel range [%f, %f]\n", minval, maxval);
        logverb("Writing output..\n");
        fprintf(fout, "P5 %i %i %i\n", nx, ny, maxpix);
        // Convert and write out chunks of "n" pixels at a time,
        // starting at "i".
        i = 0;
        while (i < (nx*ny)) {
            int n;
            n = MIN(NBUF, nx*ny - i);
            if (sixteenbit) {
                uint16_t buf[NBUF];
                for (j=0; j<n; j++)
                    buf[j] = htons(MIN(65535, MAX(0, round((img[i+j] - minval) * scale))));
                if (fwrite(buf, 2, n, fout) != n) {
                    fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
                    exit(-1);
                }
				
            } else {
                uint8_t buf[NBUF];
                for (j=0; j<n; j++)
                    buf[j] = MIN(255, MAX(0, round((img[i+j] - minval) * scale)));
                if (fwrite(buf, 1, n, fout) != n) {
                    fprintf(stderr, "Failed to write output image: %s\n", strerror(errno));
                    exit(-1);
                }
            }
            i += n;
        }
    }

    if (outfn)
        fclose(fout);
    anqfits_close(anq);
    free(img);
    logverb("Done!\n");
    return 0;
}
