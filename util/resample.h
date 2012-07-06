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

#ifndef RESAMPLE_H
#define RESAMPLE_H

struct lanczos_args_s {
	int order;
	int weighted;
};
typedef struct lanczos_args_s lanczos_args_t;

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

