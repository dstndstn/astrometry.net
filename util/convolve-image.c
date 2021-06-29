/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "convolve-image.h"
#include "mathutil.h"
#include "keywords.h"
#include "os-features.h"

float* convolve_get_gaussian_kernel_f(double sigma, double nsigma, int* p_k0, int* p_NK) {
    int K0, NK, i;
    float* kernel;

    K0 = ceil(sigma * nsigma);
    NK = 2*K0 + 1;
    kernel = malloc(NK * sizeof(float));
    for (i=0; i<NK; i++)
        kernel[i] = 1.0 / sqrt(2.0 * M_PI) / sigma *
            exp(-0.5 * square(i - K0) / square(sigma));
    if (p_k0)
        *p_k0 = K0;
    if (p_NK)
        *p_NK = NK;
    return kernel;
}


float* convolve_separable_f(const float* img, int W, int H,
                            const float* kernel, int K0, int NK,
                            float* outimg, float* tempimg) {
    return convolve_separable_weighted_f(img, W, H, NULL, kernel, K0, NK, outimg, tempimg);
}


float* convolve_separable_weighted_f(const float* img, int W, int H,
                                     const float* weight,
                                     const float* kernel, int K0, int NK,
                                     float* outimg, float* tempimg) {
    float* freeimg = NULL;
    int i, j, k;

    if (!tempimg)
        freeimg = tempimg = malloc((size_t)W * (size_t)H * sizeof(float));

    if (!outimg)
        outimg = malloc((size_t)W * (size_t)H * sizeof(float));

    for (i=0; i<H; i++) {
        /* // DEBUG
         anbool touchedleft = FALSE;
         anbool touchedright = FALSE;
         */
        for (j=0; j<W; j++) {
            float sum = 0;
            float sumw = 0;
            /*
             This is true convolution, so the kernel is flipped;
             in this loop we are adding image pixels from right to left.
             */

            /*
             printf("j=%i, W=%i, K0=%i, NK=%i, unrestricted j = [%i, %i),"
             "k -> [%i, %i), j -> [%i, %i)\n",
             j, W, K0, NK, j-0+K0, j-NK+K0, MAX(0, j + K0 - (W-1)), MIN(NK, j + K0 + 1),
             j + K0 - MAX(0, j + K0 - (W-1)), j + K0 - MIN(NK, j + K0 + 1));
             */

            for (k = MAX(0, j + K0 - (W-1));
                 k < MIN(NK, j + K0 + 1); k++) {

                /*
                 assert((j - k + K0) >= 0);
                 assert((k - k + K0) <= (W-1));
                 touchedleft |= ((j - k + K0) == 0);
                 touchedright |= ((j - k + K0) == (W-1));
                 */

                int p = i*W + j - k + K0;
                if (weight) {
                    sum  += kernel[k] * weight[p] * img[p];
                    sumw += kernel[k] * weight[p];
                } else {
                    sum  += kernel[k] * img[p];
                    sumw += kernel[k];
                }
            }
            // store into temp image in transposed order
            tempimg[j*H + i] = (sumw == 0.0) ? 0.0 : (sum / sumw);
        }
        //assert(touchedleft);
        //assert(touchedright);
    }

    for (j=0; j<W; j++) {
        for (i=0; i<H; i++) {
            float sum = 0;
            float sumw = 0;
            for (k = MAX(0, i + K0 - (H-1));
                 k < MIN(NK, i + K0 + 1); k++) {
                int p = j*H + i - k + K0;
                sum  += kernel[k] * tempimg[p];
                sumw += kernel[k];
            }
            outimg[i*W + j] = (sumw == 0.0) ? 0.0 : (sum / sumw);
        }
    }
    free(freeimg);
    return outimg;
}


