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

struct walk_token2 {
	cairo_t* cairo;
	dl* radecs;
};

static void walk_callback2(const anwcs_t* wcs, double ix, double iy, double ra, double dec, void* token) {
	struct walk_token2* walk = token;
	dl_append(walk->radecs, ra);
	dl_append(walk->radecs, dec);
}

// Returns 0 if the whole line was traced without breaks.
// Otherwise, returns the index of the point on the far side of the
// break.
static int trace_line(anwcs_t* wcs, cairo_t* cairo, dl* rd, int istart, int idir, int iend,
					  anbool firstmove) {
	int i;
	double lastra=0, lastdec=0;
	double first = TRUE;
	//logverb("trace_line: start %i, dir %i, end %i\n", istart, idir, iend);
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
	struct walk_token2 token2;
	double degstep = 1;
	dl* rd;
	int brk, end;
	dl* rd2;
	int i;

	assert(args->stepsize > 0);
	assert(args->wcs);
	assert(pargs->wcs);

	plotstuff_builtin_apply(cairo, pargs);

	logverb("Plotting outline of WCS: image size is %g x %g\n",
			anwcs_imagew(args->wcs), anwcs_imageh(args->wcs));

	token2.cairo = cairo;
	token2.radecs = dl_new(256);
	anwcs_walk_image_boundary(args->wcs, args->stepsize, walk_callback2, &token2);
	logverb("Outline: walked in %i steps\n", dl_size(token2.radecs)/2);
	rd = token2.radecs;

	// avoid special case when there is a break between
	// the beginning and end of the list.
	dl_append(rd, dl_get(rd, 0));
	dl_append(rd, dl_get(rd, 1));

	// DEBUG
	/*
	for (i=0; i<dl_size(rd)/2; i++) {
		double ra,dec,x,y, lx,ly;
		ra  = dl_get(rd, 2*i+0);
		dec = dl_get(rd, 2*i+1);
		if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y)) {
			logverb("oops!\n");
			continue;
		}
		if (i == 0) {
			lx = ly = 0;
		}

		logverb("%4i %8.1f %8.1f --> %6.1f, %6.1f  (d %g)\n", i, ra, dec, x, y, hypot(x-lx,y-ly));
		lx = x;
		ly = y;
	}
	 */

	end = dl_size(rd)/2;
	brk = trace_line(pargs->wcs, cairo, rd, 0, 1, end, TRUE);
	logdebug("tracing line 1: brk=%i\n", brk);

	if (brk) {
		int brk2;
		int brk3;
		// back out the path.
		cairo_new_path(cairo);
		// trace segment 1 backwards to 0
		brk2 = trace_line(pargs->wcs, cairo, rd, brk-1, -1, -1, TRUE);
		logdebug("traced line 1 backwards: brk2=%i\n", brk2);
		assert(brk2 == 0);

		// catch edge case: there is a break between the beginning and end of the list.
		//if (anwcs_is_discontinuous(wcs, dl_get(lastra, lastdec, ra, dec)) {
		

		// trace segment 2: from end of list backward, until we
		// hit brk2 (worst case, we [should] hit brk)
		brk2 = trace_line(pargs->wcs, cairo, rd, end-1, -1, -1, FALSE);
		logdebug("traced segment 2: brk2=%i\n", brk2);
		if (args->fill) {
			// trace segment 3: from brk2 to brk.
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
		}

		if (args->fill)
			cairo_fill(cairo);
		else
			cairo_stroke(cairo);

		// trace segments 4+5: from brk to brk2.
		if (brk2 > brk) {
			// (tracing the outline on the far side)
			brk3 = trace_line(pargs->wcs, cairo, rd, brk, 1, brk2, TRUE);
			logdebug("traced segment 4/5: brk3=%i\n", brk3);
			assert(brk3 == 0);
			// trace segment 6: from brk2 to brk.
			// (walking the discontinuity on the far side)
			if (args->fill) {
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
				cairo_close_path(cairo);
			}
		}
	}
	dl_free(token2.radecs);

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

