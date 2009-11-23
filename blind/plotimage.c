/*
  This file is part of the Astrometry.net suite.
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
#include <string.h>

#include "plotimage.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

const plotter_t plotter_image = {
	.name = "image",
	.init = plot_image_init,
	.command = plot_image_command,
	.doplot = plot_image_plot,
	.free = plot_image_free
};

void* plot_image_init(plot_args_t* plotargs) {
	plotimage_t* args = calloc(1, sizeof(plotimage_t));
	return args;
}

void plot_image_rgba_data(cairo_t* cairo, unsigned char* img, int W, int H) {
	cairo_surface_t* thissurf;
	cairo_pattern_t* pat;
	cairoutils_rgba_to_argb32(img, W, H);
	thissurf = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, W, H, W*4);
	pat = cairo_pattern_create_for_surface(thissurf);
	cairo_save(cairo);
	cairo_set_source(cairo, pat);
	cairo_paint(cairo);
	cairo_pattern_destroy(pat);
	cairo_surface_destroy(thissurf);
	cairo_restore(cairo);
}

int plot_image_read(plotimage_t* args) {
	// FIXME -- guess format from filename?
	switch (args->format) {
	case PLOTSTUFF_FORMAT_JPG:
		args->img = cairoutils_read_jpeg(args->fn, &(args->W), &(args->H));
		break;
	case PLOTSTUFF_FORMAT_PNG:
		args->img = cairoutils_read_png(args->fn, &(args->W), &(args->H));
		break;
	case PLOTSTUFF_FORMAT_PPM:
		args->img = cairoutils_read_ppm(args->fn, &(args->W), &(args->H));
		break;
	case PLOTSTUFF_FORMAT_PDF:
		ERROR("PDF format not supported");
		return -1;
	default:
		ERROR("You must set the image format with \"image_format <png|jpg|ppm>\"");
		return -1;
	}
	return 0;
}

int plot_image_set_filename(plotimage_t* args, const char* fn) {
	free(args->fn);
	args->fn = strdup_safe(fn);
	return 0;
}

int plot_image_plot(const char* command,
					cairo_t* cairo, plot_args_t* plotargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	// Plot it!
	if (!args->img) {
		if (plot_image_read(args)) {
			return -1;
		}
	}
	plot_image_rgba_data(cairo, args->img, args->W, args->H);
	// ?
	free(args->img);
	args->img = NULL;
	return 0;
}

int plot_image_setsize(plot_args_t* pargs, plotimage_t* args) {
	if (!args->img) {
		if (plot_image_read(args)) {
			return -1;
		}
	}
	pargs->W = args->W;
	pargs->H = args->H;
	return 0;
}

int plot_image_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	if (streq(cmd, "image_file")) {
		plot_image_set_filename(args, cmdargs);
	} else if (streq(cmd, "image_format")) {
		args->format = parse_image_format(cmdargs);
		if (args->format == -1)
			return -1;
	} else if (streq(cmd, "image_setsize")) {
		if (plot_image_setsize(pargs, args))
			return -1;
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_image_free(plot_args_t* plotargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	free(args->fn);
	free(args);
}

