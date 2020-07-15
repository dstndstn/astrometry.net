/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "ctmf.h"
#include "cairoutils.h"

const char* OPTIONS = "h";


int main(int argc, char *args[]) {
    int argchar;
    //char* progname = args[0];
    char *infn = NULL, *outfn = NULL;
    unsigned char* img;
    int W, H;
    unsigned char* bg;
    int i, j;
    int halfbox = 10;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
            //case '?':
            //case 'h':
            //printHelp(progname);
            //return 0;
        default:
            return -1;
        }

    if (optind != argc-2) {
        //printHelp(progname);
        exit(-1);
    }

    infn = args[optind];
    outfn = args[optind+1];

    printf("Reading input %s...\n", infn);
    img = cairoutils_read_jpeg(infn, &W, &H);
    if (!img) {
        fprintf(stderr, "Failed to read input image %s.\n", infn);
        exit(-1);
    }
    printf("Image is %i x %i.\n", W, H);

    bg = malloc(W * H * 4);

    printf("Filtering...");
    for (i=0; i<3; i++) {
        printf(" %c...", "RGB"[i]);
        fflush(stdout);
        ctmf(img + i, bg + i, W, H, 4*W, 4*W, halfbox, 4, 512*1024);
    }
    printf("\n");

    for (i=0; i<(W*H); i++)
        for (j=0; j<3; j++) {
            int ind = i*4 + j;
            if (img[ind] <= bg[ind])
                img[ind] = 0;
            else
                img[ind] -= bg[ind];
        }

    printf("Writing output...\n");
    cairoutils_write_jpeg(outfn, img, W, H);

    free(img);
    free(bg);

    printf("Done!\n");
    return 0;
}
