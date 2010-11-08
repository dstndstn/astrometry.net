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
#include <math.h>
#include <sys/param.h>

#include "plotoutline.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "starutil.h"

const plotter_t plotter_outline = {
	.name = "outline",
	.init = plot_outline_init,
	.command = plot_outline_command,
	.doplot = plot_outline_plot,
	.free = plot_outline_free
};

plotoutline_t* plot_outline_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "outline");
}

void* plot_outline_init(plot_args_t* plotargs) {
	plotoutline_t* args = calloc(1, sizeof(plotoutline_t));
	args->stepsize = 10;
	return args;
}

struct walk_token {
	cairo_t* cairo;
	bool first;
	plot_args_t* pargs;
	double lastra, lastdec;
};

static void walk_callback(const anwcs_t* wcs, double ix, double iy, double ra, double dec, void* token) {
	struct walk_token* walk = token;
	bool ok;
	double x, y;
	ok = plotstuff_radec2xy(walk->pargs, ra, dec, &x, &y);
	logverb("plotoutline: wcs x,y (%.0f,%.0f) -> RA,Dec (%.1g,%.1g) -> image x,y (%.1f, %.1f)\n",
			ix, iy, ra, dec, x, y);
	if (!ok)
		return;
	if (walk->first) {
		cairo_move_to(walk->cairo, x, y);
		walk->first = FALSE;
	} else {
		// check for wrap-around.

		if (anwcs_is_discontinuous(walk->pargs->wcs, walk->lastra, walk->lastdec, ra, dec)) {
			logmsg("plotoutline: discontinuous: (%g,%g) -- (%g,%g)\n", walk->lastra, walk->lastdec, ra, dec);
			cairo_move_to(walk->cairo, x, y);
		}

		/*
		 // scale
		 {
		 double lastx, lasty;
		 ok = plotstuff_radec2xy(walk->pargs, walk->lastra, walk->lastdec, &lastx, &lasty);
		 if (ok) {
		 double scale = arcsec_between_radecdeg(walk->lastra, walk->lastdec, ra, dec) / hypot(lastx - x, lasty - y);
		 printf("WCS scale: %g, step scale: %g\n", anwcs_pixel_scale(walk->pargs->wcs), scale);
		 }
		 }
		 */
		
		cairo_line_to(walk->cairo, x, y);
	}
	walk->lastra = ra;
	walk->lastdec = dec;
}


struct walk_token2 {
	cairo_t* cairo;
	dl* radecs;
	//plot_args_t* pargs;
	//double lastra, lastdec;
};

static void walk_callback2(const anwcs_t* wcs, double ix, double iy, double ra, double dec, void* token) {
	struct walk_token2* walk = token;
	dl_append(walk->radecs, ra);
	dl_append(walk->radecs, dec);
	//dl_append(walk->xys, ix);
	//dl_append(walk->xys, iy);
}

// Returns 0 if the whole line was traced without breaks.
// Otherwise, returns the index of the point on the far side of the
// break.
static int trace_line(anwcs_t* wcs, cairo_t* cairo, dl* rd, int istart, int idir, int iend,
					  bool firstmove) {
	int i;
	double lastra=0, lastdec=0;
	double first = TRUE;
	logverb("trace_line: start %i, dir %i, end %i\n", istart, idir, iend);
	for (i = istart; i != iend; i += idir) {
		double x,y,ra,dec;
		ra  = dl_get(rd, 2*i+0);
		dec = dl_get(rd, 2*i+1);

		//logverb("tracing: i=%i, ra,dec = %g,%g\n", i, ra, dec);

		if (anwcs_radec2pixelxy(wcs, ra, dec, &x, &y))
			// ?
			continue;

		//logverb("  x,y %g,%g\n", x, y);

		if (first) {
			if (firstmove)
				cairo_move_to(cairo, x, y);
			else
				cairo_line_to(cairo, x, y);
		} else {
			if (anwcs_is_discontinuous(wcs, lastra, lastdec, ra, dec)) {
				logmsg("plotoutline: discontinuous: (%g,%g) -- (%g,%g)\n", lastra, lastdec, ra, dec);
				return i;
			}
			cairo_line_to(cairo, x, y);
		}
		lastra = ra;
		lastdec = dec;
		first = FALSE;
	}
	return 0;
}


int plot_outline_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotoutline_t* args = (plotoutline_t*)baton;
	struct walk_token token;
	assert(args->stepsize > 0);
	assert(args->wcs);
	assert(pargs->wcs);

	plotstuff_builtin_apply(cairo, pargs);

	logmsg("Plotting outline of WCS: image size is %g x %g\n",
		   anwcs_imagew(args->wcs), anwcs_imageh(args->wcs));

	token.first = TRUE;
	token.cairo = cairo;
	token.pargs = pargs;

	//
	{
		struct walk_token2 token2;
		dl* rd;
		int brk, end;
		dl* rd2;
		double degstep;
		int i;

		token2.cairo = cairo;
		token2.radecs = dl_new(256);
		anwcs_walk_image_boundary(args->wcs, args->stepsize, walk_callback2, &token2);
		logmsg("Outline: walked in %i steps\n", dl_size(token2.radecs));
		rd = token2.radecs;

		end = dl_size(rd)/2;
		brk = trace_line(pargs->wcs, cairo, rd, 0, 1, end, TRUE);
		logverb("tracing line 1: brk=%i\n", brk);

		if (brk) {
			int brk2;
			int brk3;
			// back out the path.
			cairo_new_path(cairo);
			// trace segment 1 backwards.
			brk2 = trace_line(pargs->wcs, cairo, rd, brk-1, -1, -1, TRUE);
			logverb("traced line 1 backwards: brk2=%i\n", brk2);
			assert(brk2 == 0);
			// trace segment 2: from end of list backward, until we
			// hit brk2.
			brk2 = trace_line(pargs->wcs, cairo, rd, end-1, -1, -1, FALSE);
			logverb("traced segment 2: brk2=%i\n", brk2);
			// trace segment 3: from brk2 to brk.
			// TODO
			// anwcs_trace_discontinuity(...)?

			// 1-pixel steps.
			degstep = arcsec2deg(anwcs_pixel_scale(pargs->wcs));

			rd2 = anwcs_walk_discontinuity(pargs->wcs,
										   dl_get(rd, 2*(brk2+1)+0), dl_get(rd, 2*(brk2+1)+1),
										   dl_get(rd, 2*(brk2  )+0), dl_get(rd, 2*(brk2  )+1),
										   dl_get(rd, 2*(brk -1)+0), dl_get(rd, 2*(brk -1)+1),
										   dl_get(rd, 2*(brk   )+0), dl_get(rd, 2*(brk   )+1),
										   degstep, NULL);
			for (i=0; i<dl_size(rd2)/2; i++) {
				double x,y,ra,dec;
				ra  = dl_get(rd2, 2*i+0);
				dec = dl_get(rd2, 2*i+1);
				if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y))
					// oops.
					continue;
				cairo_line_to(cairo, x, y);
			}
			dl_free(rd2);

			cairo_close_path(cairo);

			if (args->fill)
				cairo_fill(cairo);
			else
				cairo_stroke(cairo);

			// trace segments 4+5: from brk to brk2.
			brk3 = trace_line(pargs->wcs, cairo, rd, brk, 1, brk2, TRUE);
			logverb("traced segment 4/5: brk3=%i\n", brk3);
			assert(brk3 == 0);
			// trace segment 6: from brk2 to brk.
			rd2 = anwcs_walk_discontinuity(pargs->wcs,
										   dl_get(rd, 2*(brk2  )+0), dl_get(rd, 2*(brk2  )+1),
										   dl_get(rd, 2*(brk2+1)+0), dl_get(rd, 2*(brk2+1)+1),
										   dl_get(rd, 2*(brk   )+0), dl_get(rd, 2*(brk   )+1),
										   dl_get(rd, 2*(brk -1)+0), dl_get(rd, 2*(brk -1)+1),
										   degstep, NULL);
			for (i=0; i<dl_size(rd2)/2; i++) {
				double x,y,ra,dec;
				ra  = dl_get(rd2, 2*i+0);
				dec = dl_get(rd2, 2*i+1);
				if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y))
					// oops.
					continue;
				cairo_line_to(cairo, x, y);
			}
			dl_free(rd2);
		}
		cairo_close_path(cairo);

		dl_free(token2.radecs);

	}

	/*
	 anwcs_walk_image_boundary(args->wcs, args->stepsize, walk_callback, &token);
	 cairo_close_path(cairo);
	 */
	if (args->fill)
		cairo_fill(cairo);
	else
		cairo_stroke(cairo);

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

int plot_outline_set_fill(plotoutline_t* args, bool fill) {
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

