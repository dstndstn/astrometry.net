/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009 Dustin Lang.

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
#include <sys/param.h>

#include "plotxy.h"
#include "xylist.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"

struct plotxy_args {
	char* fn;
	int ext;
	char* xcol;
	char* ycol;
	double xoff, yoff;
	int firstobj;
	int nobjs;
	double scale;
	double bglw;
	float bgr, bgg, bgb, bga;
};
typedef struct plotxy_args plotxy_t;

void* plot_xy_init(plot_args_t* plotargs) {
	plotxy_t* args = calloc(1, sizeof(plotxy_t));
	args->ext = 1;
	args->scale = 1.0;
	return args;
}

int plot_xy_command(const char* command, cairo_t* cairo,
					plot_args_t* plotargs, void* baton) {
	plotxy_t* args = (plotxy_t*)baton;

	if (streq(command, "xy")) {
		// Plot it!
		xylist_t* xyls;
		starxy_t* xy;
		int Nxy;
		int i;

		if (!args->fn) {
			ERROR("No xylist filename given");
			return -1;
		}

		// Open xylist.
		xyls = xylist_open(args->fn);
		if (!xyls) {
			ERROR("Failed to open xylist from file \"%s\"", args->fn);
			return -1;
		}
		// we don't care about FLUX and BACKGROUND columns.
		xylist_set_include_flux(xyls, FALSE);
		xylist_set_include_background(xyls, FALSE);
		if (args->xcol)
			xylist_set_xname(xyls, args->xcol);
		if (args->ycol)
			xylist_set_yname(xyls, args->ycol);

		// Find number of entries in xylist.
		xy = xylist_read_field_num(xyls, args->ext, NULL);
		xylist_close(xyls);
		if (!xy) {
			ERROR("Failed to read FITS extension %i from file %s.\n", args->ext, args->fn);
			return -1;
		}
		Nxy = starxy_n(xy);
		// If N is specified, apply it as a max.
		if (args->nobjs)
			Nxy = MIN(Nxy, args->nobjs);

		// Shift and scale xylist entries.
		if (args->xoff != 0.0 || args->yoff != 0.0) {
			for (i=0; i<Nxy; i++) {
				starxy_setx(xy, i, starxy_getx(xy, i) - args->xoff);
				starxy_sety(xy, i, starxy_gety(xy, i) - args->yoff);
			}
		}
		if (args->scale != 1.0) {
			for (i=0; i<Nxy; i++) {
				starxy_setx(xy, i, args->scale * starxy_getx(xy, i));
				starxy_sety(xy, i, args->scale * starxy_gety(xy, i));
			}
		}

		if (args->bga != 0.0) {
			// Plot background.
			cairo_save(cairo);
			if (args->bglw)
				cairo_set_line_width(cairo, args->bglw);
			else
				cairo_set_line_width(cairo, plotargs->lw + 2.0);
			cairo_set_source_rgba(cairo, args->bgr, args->bgg, args->bgb, args->bga);
			for (i=args->firstobj; i<Nxy; i++) {
				double x = starxy_getx(xy, i) + 0.5;
				double y = starxy_gety(xy, i) + 0.5;
				cairoutils_draw_marker(cairo, plotargs->marker, x, y, plotargs->markersize);
				cairo_stroke(cairo);
			}
			cairo_restore(cairo);
		}

		// Plot markers.
		for (i=args->firstobj; i<Nxy; i++) {
			double x = starxy_getx(xy, i) + 0.5;
			double y = starxy_gety(xy, i) + 0.5;
			cairoutils_draw_marker(cairo, plotargs->marker, x, y, plotargs->markersize);
			cairo_stroke(cairo);
		}
		
		return 0;
	} else {
		char* cmd;
		char* cmdargs;
		if (!split_string_once(command, " ", &cmd, &cmdargs)) {
			ERROR("Failed to split command \"%s\" into words\n", command);
			return -1;
		}
		logmsg("Command \"%s\", args \"%s\"\n", cmd, cmdargs);

		if (streq(cmd, "xy_file")) {
			args->fn = strdup(cmdargs);
		} else if (streq(cmd, "xy_ext")) {
			args->ext = atoi(cmdargs);
		} else if (streq(cmd, "xy_xcol")) {
			args->xcol = strdup(cmdargs);
		} else if (streq(cmd, "xy_ycol")) {
			args->ycol = strdup(cmdargs);
		} else if (streq(cmd, "xy_xoff")) {
			args->xoff = atof(cmdargs);
		} else if (streq(cmd, "xy_yoff")) {
			args->yoff = atof(cmdargs);
		} else if (streq(cmd, "xy_firstobj")) {
			args->firstobj = atoi(cmdargs);
		} else if (streq(cmd, "xy_nobjs")) {
			args->nobjs = atoi(cmdargs);
		} else if (streq(cmd, "xy_scale")) {
			args->scale = atof(cmdargs);
		} else if (streq(cmd, "xy_bgcolor")) {
			parse_color(cmdargs, &(args->bgr), &(args->bgg), &(args->bgb), &(args->bga));
		} else if (streq(cmd, "xy_bglw")) {
			args->bglw = atof(cmdargs);
		} else {
			ERROR("Did not understand command \"%s\"", cmd);
			return -1;
		}
		return 0;
	}	
}

void plot_xy_free(plot_args_t* plotargs, void* baton) {
	plotxy_t* args = (plotxy_t*)baton;
	free(args->xcol);
	free(args->ycol);
	free(args->fn);
	free(args);
}

