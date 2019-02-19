/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"
#include "convolve-image.h"
#include "fitsioutils.h"

void test_conv_1(CuTest* tc) {
    int W = 13;
    int H = 11;
    int i;
    float* img = malloc(W * H * sizeof(float));
    float* kernel;
    float* cimg;
    int K0, NK;
    double sig = 1.0;

    for (i=0; i<(W*H); i++)
        img[i] = 0;
    img[(H/2)*W + (W/2)] = 1.;

    kernel = convolve_get_gaussian_kernel_f(sig, 5., &K0, &NK);
    cimg = convolve_separable_f(img, W, H, kernel, K0, NK, NULL, NULL);

    fits_write_float_image(cimg, W, H, "test-conv.fits");

}


