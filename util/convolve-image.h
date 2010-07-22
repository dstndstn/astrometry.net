

float* convolve_get_gaussian_kernel_f(double sigma, double nsigma, int* k0, int* NK);

float* convolve_1d_f(const float* img, int W, int H,
					 const float* kernel, int k0, int NK,
					 float* outimg, float* tempimg);

