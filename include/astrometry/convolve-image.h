/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef CONVOLVE_IMAGE_H
#define CONVOLVE_IMAGE_H


float* convolve_get_gaussian_kernel_f(double sigma, double nsigma, int* k0, int* NK);

// Does 2-D convolution by applying the same kernel in x and y directions.
// Output image can be the same as input image.
float* convolve_separable_f(const float* img, int W, int H,
                            const float* kernel, int k0, int NK,
                            float* outimg, float* tempimg);

/** Hacky -- "weight" must be the same size as "img".  Weights pixels
 by the kernel and the weight image.  Probably only useful/correct if the
 weight matrix contains a constant and zeros for masked pixels.
 */
float* convolve_separable_weighted_f(const float* img, int W, int H,
                                     const float* weight,
                                     const float* kernel, int k0, int NK,
                                     float* outimg, float* tempimg);

#endif
