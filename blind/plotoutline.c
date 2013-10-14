/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2010, 2012 Dustin Lang.

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
#include <math.h>
#include <sys/param.h>
#include <assert.h>

#include "plotoutline.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "starutil.h"

DEFINE_PLOTTER(outline);

plotoutline_t* plot_outline_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "outline");
}

void* plot_outline_init(plot_args_t* plotargs) {
	plotoutline_t* args = calloc(1, sizeof(plotoutline_t));
	args->stepsize = 10;
	return args;
}

static void walk_callback(const anwcs_t* wcs, double ix, double iy,
                          double ra, double dec, void* token) {
    dl* radecs = (dl*)token;
	dl_append(radecs, ra);
	dl_append(radecs, dec);
}

int plot_outline_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	dl* rd;
	int i;
    pl* lists;

	assert(args->stepsize > 0);
	assert(args->wcs);
	assert(pargs->wcs);

	plotstuff_builtin_apply(cairo, pargs);

	logverb("Plotting outline of WCS: image size is %g x %g\n",
			anwcs_imagew(args->wcs), anwcs_imageh(args->wcs));

    rd = dl_new(256);
	anwcs_walk_image_boundary(args->wcs, args->stepsize, walk_callback, rd);
	logverb("Outline: walked in %i steps\n", dl_size(rd)/2);

    if (dl_size(rd) == 0) {
        printf("plot_outline: empty WCS outline.\n");
        anwcs_print(args->wcs, stdout);
        dl_free(rd);
        return 0;
    }

	// avoid special case when there is a break between
	// the beginning and end of the list.
	dl_append(rd, dl_get(rd, 0));
	dl_append(rd, dl_get(rd, 1));

    lists = anwcs_walk_outline(pargs->wcs, rd, args->fill);
    dl_free(rd);
    for (i=0; i<pl_size(lists); i++) {
        dl* xy = pl_get(lists, i);
        int j;
        for (j=0; j<dl_size(xy)/2; j++) {
            double x,y;
            x = dl_get(xy, j*2+0);
            y = dl_get(xy, j*2+1);
            if (j == 0) {
                cairo_move_to(cairo, x, y);
            } else {
                cairo_line_to(cairo, x, y);
            }
        }
        cairo_close_path(cairo);
		if (args->fill)
			cairo_fill(cairo);
		else
			cairo_stroke(cairo);
        dl_free(xy);
    }
    pl_free(lists);
	return 0;
}

int plot_outline_set_wcs_size(plotoutline_t* args, int W, int H) {
  if (!args->wcs) {
    ERROR("No WCS is currently set.");
    return -1;
  }
  anwcs_set_size(args->wcs, W, H);
  return 0;
}

int plot_outline_set_wcs_file(plotoutline_t* args, const char* filename, int ext) {
	anwcs_t* wcs = anwcs_open(filename, ext);
	if (!wcs) {
		ERROR("Failed to read WCS file \"%s\"", filename);
		return -1;
	}
	logverb("Read WCS file %s\n", filename);
	if (args->wcs)
		anwcs_free(args->wcs);
	args->wcs = wcs;
	//anwcs_print(args->wcs, stdout);
	return 0;
}

int plot_outline_set_wcs(plotoutline_t* args, sip_t* wcs) {
	if (args->wcs)
		anwcs_free(args->wcs);
	args->wcs = anwcs_new_sip(wcs);
	return 0;
}

int plot_outline_set_fill(plotoutline_t* args, anbool fill) {
	args->fill = fill;
	return 0;
}

int plot_outline_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	if (streq(cmd, "outline_wcs")) {
		if (plot_outline_set_wcs_file(args, cmdargs, 0)) {
			return -1;
		}
	} else if (streq(cmd, "outline_fill")) {
		if (streq(cmdargs, "0"))
			args->fill = FALSE;
		else
			args->fill = TRUE;
	} else if (streq(cmd, "outline_step")) {
		args->stepsize = atof(cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_outline_free(plot_args_t* plotargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	free(args);
}

