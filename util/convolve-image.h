/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.

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
