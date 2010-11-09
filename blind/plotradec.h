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

#ifndef PLOTRADEC_H
#define PLOTRADEC_H

#include "plotstuff.h"

struct plotradec_args {
	char* fn;
	int ext;
	char* racol;
	char* deccol;
	int firstobj;
	int nobjs;

	// coordinates added with radec_val <x> <y>
	dl* radecvals;
};
typedef struct plotradec_args plotradec_t;

plotradec_t* plot_radec_get(plot_args_t* pargs);

void plot_radec_reset(plotradec_t* args);

// Called prior to cairo surface initialization.
void* plot_radec_init(plot_args_t* args);

void plot_radec_set_racol(plotradec_t* args, const char* col);
void plot_radec_set_deccol(plotradec_t* args, const char* col);
void plot_radec_set_filename(plotradec_t* args, const char* fn);

int plot_radec_command(const char* command, const char* cmdargs,
					plot_args_t* args, void* baton);

int plot_radec_count_inbounds(plot_args_t* pargs, plotradec_t* args);

int plot_radec_plot(const char* command, cairo_t* cairo,
				 plot_args_t* plotargs, void* baton);

void plot_radec_free(plot_args_t* args, void* baton);

void plot_radec_vals(plotradec_t* args, double ra, double dec);

//extern const plotter_t plotter_radec;
DECLARE_PLOTTER(radec);

#endif
