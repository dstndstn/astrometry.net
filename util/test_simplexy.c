/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cutest.h"
#include "dimage.h"
#include "simplexy.h"
#include "log.h"

/**
 > python -c "from pylab import *; I=imread('test_dcen3x3_1.pgm'); print ','.join(['%i'%x for x in I.ravel()]); print I.shape"
 */
void test_simplexy_1(CuTest* tc) {
    float image[] = {
        36,40,47,53,30,29,31,36,32,29,26,22,11,11,11,35,32,27,32,51,50,46,41,36,32,31,26,11,11,11,15,15,24,43,49,51,47,41,37,37,31,17,11,11,11,21,28,19,47,51,54,68,81,82,64,43,41,30,27,21,11,16,47,80,120,138,175,199,188,143,81,43,35,31,27,17,48,114,152,159,163,182,192,174,135,81,41,31,28,24,14,44,108,142,127,108,98,90,70,56,42,30,23,20,16,13,22,47,56,67,48,44,42,29,29,26,11,16,12,12,36,40,34,30,30,24,31,34,27,27,19,11,14,14,11,53,51,36,31,24,19,20,11,11,11,11,11,14,16,14,55,50,37,34,21,28,28,19,17,14,11,16,11,12,14
    };
    int H = 11;
    int W = 15;
    int rtn;
    float x,y;
    int N;
    int* objects;
    float sigma = 20;
    int i;

    log_init(LOG_ALL);

    for (i=0;; i++) {
        float sigma = 10.0;
        float G = exp(-((float)(i*i) / (2.0*sigma*sigma)));
        debug("Nsigma %g: G=%g\n", (float)i/sigma, G);
        if (G == 0)
            break;
    }

    CuAssertIntEquals(tc, W*H, sizeof(image)/sizeof(float));

    objects = malloc(W*H*sizeof(int));
    for (i=0; i<W*H; i++)
        objects[i] = 0;

    rtn = dallpeaks(image, W, H, objects, &x, &y, &N,
                    SIMPLEXY_DEFAULT_DPSF, sigma, SIMPLEXY_DEFAULT_DLIM,
                    SIMPLEXY_DEFAULT_SADDLE, 1, 1, 1,
                    SIMPLEXY_DEFAULT_MAXSIZE);
    CuAssertIntEquals(tc, 1, rtn);
    CuAssertIntEquals(tc, 1, N);
}
