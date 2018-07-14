/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef RESAMPLE_H
#define RESAMPLE_H

typedef struct {
    int order;
    int weighted;
} lanczos_args_t;

/***
 All the lanczos_* functions take a "lanczos_args_t*". for their "void* token".

 They're declared this way for ease of generic use as callbacks
 (eg in coadd.c)
 */

double lanczos(double x, int order);

double nearest_resample_f(double px, double py, const float* img,
                          const float* weightimg, int W, int H,
                          double* out_wt, void* token);

double lanczos_resample_f(double px, double py,
                          const float* img, const float* weightimg,
                          int W, int H, double* out_wt, void* token);

double lanczos_resample_unw_sep_f(double px, double py,
                                  const float* img,
                                  int W, int H, void* token);

double nearest_resample_d(double px, double py, const double* img,
                          const double* weightimg, int W, int H,
                          double* out_wt, void* token);

double lanczos_resample_d(double px, double py,
                          const double* img, const double* weightimg,
                          int W, int H, double* out_wt, void* token);


#endif

