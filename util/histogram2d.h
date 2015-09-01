/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
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
