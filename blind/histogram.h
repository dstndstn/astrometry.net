/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <stdio.h>

struct histogram {
	double min;
	double binsize;
	int Nbins;
	int* hist;
};
typedef struct histogram histogram;

histogram* histogram_new_nbins(double minimum, double maximum, int Nbins);

histogram* histogram_new_binsize(double minimum, double maximum, double binsize);

void histogram_free(histogram* h);

int histogram_add(histogram* h, double val);

void histogram_print_matlab(histogram* h, FILE* fid);

void histogram_print_matlab_bin_centers(histogram* h, FILE* fid);

// assumes each count is on the left side of the bin.
double histogram_mean(histogram* h);

#endif
