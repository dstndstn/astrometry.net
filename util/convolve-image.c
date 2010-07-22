#include <stdlib.h>
#include <math.h>
#include <sys/param.h>

#include "convolve-image.h"
#include "mathutil.h"
#include "keywords.h"

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


float* convolve_1d_f(const float* img, int W, int H,
				   const float* kernel, int K0, int NK,
				   float* outimg, float* tempimg) {
	float* freeimg = NULL;
	int i, j, k;

	if (!tempimg)
		freeimg = tempimg = malloc(W * H * sizeof(float));

	if (!outimg)
		outimg = malloc(W * H * sizeof(float));

	for (i=0; i<H; i++) {
		for (j=0; j<W; j++) {
			float sum = 0;
			for (k=0; k<NK; k++)
				sum += kernel[k] * img[i*W + MIN(W-1, MAX(0, j - k + K0))];
			// store into temp image in transposed order
			tempimg[j*H + i] = sum;
		}
	}
	for (j=0; j<W; j++) {
		for (i=0; i<H; i++) {
			float sum = 0;
			for (k=0; k<NK; k++)
				sum += kernel[k] * tempimg[j*H + MIN(H-1, MAX(0, i - k + K0))];
			outimg[i*W + j] = sum;
		}
	}
	free(freeimg);
	return outimg;
}


