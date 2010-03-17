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

#include "plotgrid.h"
#include "sip-utils.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

const plotter_t plotter_grid = {
	.name = "grid",
	.init = plot_grid_init,
	.command = plot_grid_command,
	.doplot = plot_grid_plot,
	.free = plot_grid_free
};

void* plot_grid_init(plot_args_t* plotargs) {
	plotgrid_t* args = calloc(1, sizeof(plotgrid_t));
	args->dolabel = TRUE;
	return args;
}

static void pretty_label(double x, char* buf) {
	int i;
	sprintf(buf, "%.2f", x);
	logverb("label: \"%s\"\n", buf);
	// Look for decimal point.
	if (!strchr(buf, '.')) {
		logverb("no decimal point\n");
		return;
	}
	// Trim trailing zeroes (after the decimal point)
	i = strlen(buf)-1;
	while (buf[i] == '0') {
		buf[i] = '\0';
		logverb("trimming trailing zero at %i: \"%s\"\n", i, buf);
		i--;
		assert(i > 0);
	}
	// Trim trailing decimal point, if it exists.
	i = strlen(buf)-1;
	if (buf[i] == '.') {
		buf[i] = '\0';
		logverb("trimming trailing decimal point at %i: \"%s\"\n", i, buf);
	}
}

static void add_text(cairo_t* cairo, double x, double y,
					 const char* txt, plot_args_t* pargs,
					 float* bgrgba,
					 float* width, float* height) {
	float ex,ey;
	float l,r,t,b;
	cairo_text_extents_t ext;
	double textmargin = 2.0;
	cairo_text_extents(cairo, txt, &ext);
	// x center
	x -= (ext.width + ext.x_bearing)/2.0;
	// y center
	y -= ext.y_bearing/2.0;

	l = x + ext.x_bearing;
	r = l + ext.width;
	t = y + ext.y_bearing;
	b = t + ext.height;
	l -= textmargin;
	r += (textmargin + 1);
	t -= textmargin;
	b += (textmargin + 1);

	// Move away from edges...
	ex = ey = 0.0;
	if (l < 0)
		ex = -l;
	if (t < 0)
		ey = -t;
	if (r > pargs->W)
		ex = -(r - pargs->W);
	if (b > pargs->H)
		ey = -(b - pargs->H);
	x += ex;
	l += ex;
	r += ex;
	y += ey;
	t += ey;
	b += ey;

	cairo_save(cairo);
	// blank out underneath the text...
	if (bgrgba) {
		cairo_set_rgba(cairo, bgrgba);
		cairo_move_to(cairo, l, t);
		cairo_line_to(cairo, l, b);
		cairo_line_to(cairo, r, b);
		cairo_line_to(cairo, r, t);
		cairo_close_path(cairo);
		cairo_fill(cairo);
	}
	cairo_restore(cairo);

	cairo_move_to(cairo, x, y);
	cairo_show_text(cairo, txt);

	if (width)
		*width = (r - l);
	if (height)
		*height = (b - t);
}

int plot_grid_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotgrid_t* args = (plotgrid_t*)baton;
	double ramin,ramax,decmin,decmax;
	double ra,dec;
	float bgrgba[4] = { 0,0,0,1 };

	if (!pargs->wcs) {
		ERROR("No WCS was set -- can't plot grid lines");
		return -1;
	}
	// Find image bounds in RA,Dec...
	sip_get_radec_bounds(pargs->wcs, 50, &ramin, &ramax, &decmin, &decmax);
	if (args->rastep == 0 || args->decstep == 0) {
		// FIXME -- choose defaults
		ERROR("Need grid_rastep, grid_decstep");
		return -1;
	}
	logverb("Image bounds: RA %g, %g, Dec %g, %g\n",
			ramin, ramax, decmin, decmax);
	for (ra = args->rastep * floor(ramin / args->rastep);
		 ra <= args->rastep * ceil(ramax / args->rastep);
		 ra += args->rastep) {
		plot_line_constant_ra(pargs, ra, decmin, decmax);
		cairo_stroke(pargs->cairo);
	}
	for (dec = args->decstep * floor(decmin / args->decstep);
		 dec <= args->decstep * ceil(decmax / args->decstep);
		 dec += args->decstep) {
		plot_line_constant_dec(pargs, dec, ramin, ramax);
		cairo_stroke(pargs->cairo);
	}

	//logmsg("Dolabel: %i\n", (int)args->dolabel);
	args->dolabel = (args->ralabelstep > 0) && (args->declabelstep > 0);
	if (args->dolabel) {
		double cra, cdec;
		if (args->ralabelstep == 0 || args->declabelstep == 0) {
			// FIXME -- choose defaults
			ERROR("Need grid_ralabelstep, grid_declabelstep");
			return -1;
		}
		logmsg("Adding grid labels...\n");
		sip_get_radec_center(pargs->wcs, &cra, &cdec);
		assert(cra >= ramin && cra <= ramax);
		assert(cdec >= decmin && cdec <= decmax);
		for (ra = args->ralabelstep * floor(ramin / args->ralabelstep);
			 ra <= args->ralabelstep * ceil(ramax / args->ralabelstep);
			 ra += args->ralabelstep) {
			double out;
			double in = cdec;
			char label[32];
			double x,y;
			bool ok;
			int i, N;
			double lra;
			bool gotit;
			int dir;
			logverb("Labelling RA=%g\n", ra);
			// where does this line leave the image?
			// cdec is inside; take steps away until we go outside.
			gotit = FALSE;
			// dir is first 1, then -1.
			for (dir=1; dir>=-1; dir-=2) {
				for (i=1;; i++) {
					// take 10-deg steps
					out = cdec + i*dir*10.0;
					if (out >= 100.0 || out <= -100)
						break;
					out = MIN(90, MAX(-90, out));
					logverb("dec in=%g, out=%g\n", in, out);
					if (!sip_is_inside_image(pargs->wcs, ra, out)) {
						gotit = TRUE;
						break;
					}
				}
				if (gotit)
					break;
			}
			if (!gotit) {
				ERROR("Couldn't find a Dec outside the image for RA=%g\n", ra);
				continue;
			}

			i=0;
			N = 10;
			while (!sip_is_inside_image(pargs->wcs, ra, in)) {
				if (i == N)
					break;
				in = decmin + (double)i/(double)(N-1) * (decmax-decmin);
				i++;
			}
			if (!sip_is_inside_image(pargs->wcs, ra, in))
				continue;
			while (fabs(out - in) > 1e-6) {
				// hahaha
				double half;
				bool isin;
				half = (out + in) / 2.0;
				isin = sip_is_inside_image(pargs->wcs, ra, half);
				if (isin)
					in = half;
				else
					out = half;
			}
			lra = ra;
			if (lra < 0)
				lra += 360;
			if (lra >= 360)
				lra -= 360;
			//snprintf(label, sizeof(label), "%.1f", lra);
			pretty_label(lra, label);
			logmsg("Label \"%s\" at (%g,%g)\n", label, ra, in);
			ok = sip_radec2pixelxy(pargs->wcs, ra, in, &x, &y);

			add_text(pargs->cairo, x, y, label, pargs, bgrgba, NULL, NULL);
		}
		for (dec = args->declabelstep * floor(decmin / args->declabelstep);
			 dec <= args->declabelstep * ceil(decmax / args->declabelstep);
			 dec += args->declabelstep) {
			double out;
			double in = cra;
			char label[32];
			double x,y;
			bool ok;
			int i, N;
			bool gotit;
			int dir;
			logverb("Labelling Dec=%g\n", dec);
			gotit = FALSE;
			// dir is first 1, then -1.
			for (dir=1; dir>=-1; dir-=2) {
				for (i=1;; i++) {
					// take 10-deg steps
					out = cra + i*dir*10.0;
					if (out > 370.0 || out <= -10)
						break;
					out = MIN(360, MAX(0, out));
					logverb("ra in=%g, out=%g\n", in, out);
					if (!sip_is_inside_image(pargs->wcs, out, dec)) {
						gotit = TRUE;
						break;
					}
				}
				if (gotit)
					break;
			}
			if (!gotit) {
				ERROR("Couldn't find an RA outside the image for Dec=%g\n", dec);
				continue;
			}
			i=0;
			N = 10;
			while (!sip_is_inside_image(pargs->wcs, in, dec)) {
				if (i == N)
					break;
				in = ramin + (double)i/(double)(N-1) * (ramax-ramin);
				i++;
			}
			if (!sip_is_inside_image(pargs->wcs, in, dec))
				continue;
			while (fabs(out - in) > 1e-6) {
				// hahaha
				double half;
				bool isin;
				half = (out + in) / 2.0;
				isin = sip_is_inside_image(pargs->wcs, half, dec);
				if (isin)
					in = half;
				else
					out = half;
			}
			//snprintf(label, sizeof(label), "%.1f", dec);
			pretty_label(dec, label);
			logmsg("Label Dec=\"%s\" at (%g,%g)\n", label, in, dec);
			ok = sip_radec2pixelxy(pargs->wcs, in, dec, &x, &y);

			add_text(pargs->cairo, x, y, label, pargs, bgrgba, NULL, NULL);
		}
		
		
	}


	return 0;
}

int plot_grid_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotgrid_t* args = (plotgrid_t*)baton;
	if (streq(cmd, "grid_rastep")) {
		args->rastep = atof(cmdargs);
	} else if (streq(cmd, "grid_decstep")) {
		args->decstep = atof(cmdargs);
	} else if (streq(cmd, "grid_ralabelstep")) {
		args->ralabelstep = atof(cmdargs);
	} else if (streq(cmd, "grid_declabelstep")) {
		args->declabelstep = atof(cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_grid_free(plot_args_t* plotargs, void* baton) {
	plotgrid_t* args = (plotgrid_t*)baton;
	free(args);
}

