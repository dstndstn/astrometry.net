/*
   This file is part of the Astrometry.net suite.
   Copyright 2007 Dustin Lang.

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
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>
#include <string.h>

#include "an-bool.h"
#include "ioutils.h"
#include "cairoutils.h"

const char* OPTIONS = "f";

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int argchar;
    int I;
    bool fullsize = FALSE;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 'f':
            fullsize = TRUE;
            break;
        }

    for (I=optind; I<argc; I++) {
        char* fn;
        int W, H;
        unsigned char* img;
        int halfW, halfH;
        unsigned char* halfimg;
        int i, j;
        int z;
        char* dot;
        char* basefn;
        bool jpeg, png;
        char* outfn;

        fn = args[I];
        dot = strrchr(fn, '.');
        if (!dot) {
            fprintf(stderr, "filename %s has no suffix.\n", fn);
            continue;
        }
        jpeg = png = FALSE;
        if (!strcasecmp(dot+1, "jpg") ||
            !strcasecmp(dot+1, "jpeg")) {
            jpeg = TRUE;
        } else if (!strcasecmp(dot+1, "png")) {
            png = TRUE;
        } else {
            fprintf(stderr, "filename %s doesn't end with jpeg or png.\n", fn);
            continue;
        }
        asprintf_safe(&basefn, "%.*s", dot - fn, fn);

        img = NULL;
        if (jpeg) {
            img = cairoutils_read_jpeg(fn, &W, &H);
        } else {
            img = cairoutils_read_png(fn, &W, &H);
        }
        if (!img) {
            fprintf(stderr, "Failed to read image from file %s.\n", fn);
            exit(-1);
        }

        if (!jpeg && fullsize) {
            asprintf_safe(&outfn, "%s.jpg", basefn);
            printf("Writing file %s: %i x %i.\n", outfn, W, H);
            if (cairoutils_write_jpeg(outfn, halfimg, W, H)) {
                fprintf(stderr, "Failed to write JPEG output %s.\n", outfn);
                exit(-1);
            }
            free(outfn);
        }

        for (z=1;; z++) {
            halfW = (W + 1) / 2;
            halfH = (H + 1) / 2;
            halfimg = malloc(4 * halfW * halfH);
            for (j=0; j<halfH; j++) {
                for (i=0; i<halfW; i++) {
                    int ii, jj;
                    int np = 0;
                    int rsum=0, gsum=0, bsum=0;
                    for (jj=0; jj<2; jj++) {
                        for (ii=0; ii<2; ii++) {
                            int px, py;
                            px = i*2 + ii;
                            py = j*2 + jj;
                            if ((py < H) && (px < W)) {
                                np++;
                                rsum += img[4*(py*W + px) + 0];
                                gsum += img[4*(py*W + px) + 1];
                                bsum += img[4*(py*W + px) + 2];
                            } 
                       }
                    }
                    halfimg[4*(j*halfW + i) + 0] = (int)rintf(rsum/(float)np);
                    halfimg[4*(j*halfW + i) + 1] = (int)rintf(gsum/(float)np);
                    halfimg[4*(j*halfW + i) + 2] = (int)rintf(bsum/(float)np);
                    halfimg[4*(j*halfW + i) + 3] = 255;
                }
            }

            asprintf_safe(&outfn, "%s-%i.jpg", basefn, z);

            printf("Writing file %s: %i x %i.\n", outfn, halfW, halfH);

            if (cairoutils_write_jpeg(outfn, halfimg, halfW, halfH)) {
                fprintf(stderr, "Failed to write JPEG output %s.\n", outfn);
                exit(-1);
            }
            free(outfn);

            free(img);
            img = halfimg;
            W = halfW;
            H = halfH;

            if (halfW ==1 && halfH == 1)
                break;
        }
        free(img);
        free(basefn);
    }

    return 0;
}
