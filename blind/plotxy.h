/*
 This file is part of the Astrometry.net suite.
 Copyright 2009, 2010 Dustin Lang.

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

#ifndef PLOTXY_H
#define PLOTXY_H

#include "plotstuff.h"

struct plotxy_args {
	char* fn;
	int ext;
	char* xcol;
	char* ycol;
	double xoff, yoff;
	int firstobj;
	int nobjs;
	double scale;

	// coordinates added with xy_val <x> <y>
	dl* xyvals;

	// if WCS is set, x,y are treated as FITS pixel coords;
	// that is, this are pushed through the WCS unmodified, then the resulting
	// RA,Dec is pushed through the plot WCS, producing FITS coords, from which
	// 1,1 is subtracted to yield 0-indexed image coords.
	anwcs_t* wcs;
};
typedef struct plotxy_args plotxy_t;

plotxy_t* plot_xy_get(plot_args_t* pargs);

// Called prior to cairo surface initialization.
void* plot_xy_init(plot_args_t* args);

// Set the plot size based on IMAGEW,IMAGEH in the xylist header.
int plot_xy_setsize(plot_args_t* args, plotxy_t* xyargs);

// Clears the list of points.
void plot_xy_clear_list(plotxy_t* args);

void plot_xy_set_xcol(plotxy_t* args, const char* col);
void plot_xy_set_ycol(plotxy_t* args, const char* col);
void plot_xy_set_filename(plotxy_t* args, const char* fn);
int plot_xy_set_wcs_filename(plotxy_t* args, const char* fn, int ext);
int plot_xy_set_offsets(plotxy_t* args, double xo, double yo);

int plot_xy_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_xy_plot(const char* command, cairo_t* cairo,
				 plot_args_t* plotargs, void* baton);

void plot_xy_free(plot_args_t* args, void* baton);

void plot_xy_vals(plotxy_t* args, double x, double y);

DECLARE_PLOTTER(xy);

#endif
