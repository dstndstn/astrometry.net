/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef HISTOGRAM2D_H
#define HISTOGRAM2D_H

#include <stdio.h>

#define HIST2D_TRUNCATE 0
#define HIST2D_DISCARD  1

#define HIST2D_DISCARDED_X -1
#define HIST2D_DISCARDED_Y -2

struct histogram2d {
	double minx, miny;
	double binsizex, binsizey;
	int NX, NY;
	int* hist;
	// edge-handling policies HIST2D_{TRUNCATE, DISCARD, ...}
	int edgex, edgey;
};
typedef struct histogram2d histogram2d;

histogram2d* histogram2d_new_nbins(double minX, double maxX, int NbinsX,
								   double minY, double maxY, int NbinsY);

void histogram2d_set_x_edges(histogram2d* h, int edgepolicy);
void histogram2d_set_y_edges(histogram2d* h, int edgepolicy);

void histogram2d_free(histogram2d* h);

int histogram2d_add(histogram2d* h, double x, double y);

#endif
