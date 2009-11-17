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

struct plotimage_args {
	char* fn;
	char* format;
	// FIXME -- alpha?
};
typedef struct plotimage_args plotimage_t;

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

int plot_image_command(const char* command, cairo_t* cairo,
					   plot_args_t* plotargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;

	if (streq(command, "image")) {
		// Plot it!
		unsigned char* img = NULL;
		int W, H;

		// FIXME -- guess format from filename.
		if (streq(args->format, "png")) {
			img = cairoutils_read_png(args->fn, &W, &H);
		} else if (streq(args->format, "jpg")) {
			img = cairoutils_read_jpeg(args->fn, &W, &H);
		} else if (streq(args->format, "ppm")) {
			img = cairoutils_read_ppm(args->fn, &W, &H);
		} else {
			ERROR("You must set the image format with \"image_format <png|jpg|ppm>\"");
			return -1;
		}
		plot_image_rgba_data(cairo, img, W, H);
		free(img);

	} else {
		char* cmd;
		char* cmdargs;
		if (!split_string_once(command, " ", &cmd, &cmdargs)) {
			ERROR("Failed to split command \"%s\" into words\n", command);
			return -1;
		}
		logmsg("Command \"%s\", args \"%s\"\n", cmd, cmdargs);

		if (streq(cmd, "image_file")) {
			free(args->fn);
			args->fn = strdup(cmdargs);
		} else if (streq(cmd, "image_format")) {
			free(args->format);
			args->format = strdup(cmdargs);
		} else {
			ERROR("Did not understand command \"%s\"", cmd);
			return -1;
		}
	}
	return 0;
}

void plot_image_free(plot_args_t* plotargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	free(args->fn);
	free(args->format);
	free(args);
}

