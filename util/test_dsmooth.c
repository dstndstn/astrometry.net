/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cutest.h"

int dsmooth(float* image, int nx, int ny, float sigma, float* smooth);
void dsmooth2(float* image, int nx, int ny, float sigma, float* smooth);

int compare_images(float *i1, float* i2, int nx, int ny, float eps) {
    int i, j;
    int fail = 0;
    for (j=0; j<ny; j++) {
        for (i=0; i<nx; i++) {
            int ind = nx*j+i;
            float v1 = i1[ind];
            float v2 = i2[ind];
            if (fabsf(v1 - v2) > eps) {
				printf("failure -- %f != %f (delta %f)\n", v1, v2, fabsf(v1-v2));
				fail++;
			}
        }
	}
	return fail;
}

float* random_image(int nx, int ny) {
    int i;
    float* img;
    img = malloc(nx * ny * sizeof(float));
    for (i=0; i<(nx*ny); i++)
        img[i] = rand() / (float)RAND_MAX;
    return img;
}

void test_dsmooth_vs_dsmooth2(CuTest* tc) {
    float* img;
    float* img_orig;
    int nx, ny;
    float* smooth1;
    float* smooth2;
    float sigma;
    int bites;
    float eps;

    nx = 20;
    ny = 19;
    sigma = 2.0;
    eps = 1e-6;

    bites = nx * ny * sizeof(float);

    img = random_image(nx, ny);
    img_orig = calloc(bites, 1);
    memcpy(img_orig, img, bites);

	CuAssertIntEquals(tc, 0, compare_images(img, img_orig, nx, ny, 0.0));

    smooth1 = calloc(bites, 1);
    smooth2 = calloc(bites, 1);

    dsmooth(img, nx, ny, sigma, smooth1);
    // test: don't change the input image
	CuAssertIntEquals(tc, 0, compare_images(img, img_orig, nx, ny, 0.0));
    dsmooth2(img, nx, ny, sigma, smooth2);
    // test: don't change the input image
	CuAssertIntEquals(tc, 0, compare_images(img, img_orig, nx, ny, 0.0));

    // test: dsmooth == dsmooth2
	CuAssertIntEquals(tc, 0, compare_images(smooth1, smooth2, nx, ny, eps));

    free(img);
    free(img_orig);
    free(smooth1);
    free(smooth2);
}

void test_dsmooth2_inplace(CuTest* tc) {
    float* img;
    int nx, ny;
    float* smooth1;
    float* smooth2;
    float sigma;
    int bites;
    float eps;

    nx = 20;
    ny = 19;
    sigma = 2.0;
    eps = 1e-6;

    bites = nx * ny * sizeof(float);

    img = random_image(nx, ny);
    smooth1 = calloc(bites, 1);
    smooth2 = calloc(bites, 1);

    dsmooth2(img, nx, ny, sigma, smooth2);

    // test: can we smooth in-place with dsmooth2?
    memcpy(smooth1, img, bites);
	CuAssertIntEquals(tc, 0, compare_images(img, smooth1, nx, ny, 0.0));

    dsmooth2(smooth1, nx, ny, sigma, smooth1);
	CuAssertIntEquals(tc, 0, compare_images(smooth1, smooth2, nx, ny, eps));

    free(img);
    free(smooth1);
    free(smooth2);
}
