/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include "cairoutils.h"

int main() {
    unsigned char img[4];
    int W=1;
    int H=1;

    float x=20;
    float mag=100;
    float back=0;

    img[0] = 0;
    img[1] = 0;
    img[2] = 0;
    img[3] = 128;
    cairoutils_stream_png(stdout, img, W, H);

    return 0;
}
