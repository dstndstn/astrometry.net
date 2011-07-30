




struct lanczos_args_s {
	int order;
};
typedef struct lanczos_args_s lanczos_args_t;


double nearest_resample_f(double px, double py, const float* img,
						  const float* weightimg, int W, int H,
						  double* out_wt, void* token);

double lanczos_resample_f(double px, double py,
						  const float* img, const float* weightimg,
						  int W, int H, double* out_wt, void* token);

double nearest_resample_d(double px, double py, const double* img,
						  const double* weightimg, int W, int H,
						  double* out_wt, void* token);

double lanczos_resample_d(double px, double py,
						  const double* img, const double* weightimg,
						  int W, int H, double* out_wt, void* token);

