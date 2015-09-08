/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
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
