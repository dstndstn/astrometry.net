/*
   This file is part of the Astrometry.net suite.
   Copyright 2009, Dustin Lang.

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
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <sys/param.h>

#include <cairo.h>

#include "tilerender.h"
#include "render_boundary.h"
#include "sip_qfits.h"
#include "cairoutils.h"
#include "ioutils.h"

// Ouch
const char* wcs_dirs[] = {
	"/home/dstn/test/web-data",
	/* 
	 "/home/gmaps/ontheweb-data",
	 "/home/gmaps/test/web-data",
	 "/home/gmaps/apod-solves",
	 */
	"."
};

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_boundary: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

int render_boundary(cairo_t* cairo, render_args_t* args) {
	int i, I;
	double lw = args->linewidth;
	double r, g, b;
	double alpha;
	bool fill = FALSE;
	dl* dargs = dl_new(4);
	int nsteps = 20;

	r = b = 0;
    g = 1.0;
	alpha = 1.0;

	logmsg("Starting.\n");

	cairo_set_line_width(cairo, lw);
	cairo_set_source_rgba(cairo, r, g, b, alpha);

	for (I=0; I<sl_size(args->arglist); I++) {
		char* arg = sl_get(args->arglist, I);
		if (starts_with(arg, "bwcsfn ")) {
			char* fn;
			char* wcsfn;
			sip_t* res;
			sip_t wcs;
			int W, H;
			fn = arg + strlen("bwcsfn ");
			wcsfn = find_file_in_dirs(wcs_dirs, sizeof(wcs_dirs)/sizeof(char*), fn, TRUE);
			if (!wcsfn) {
				logmsg("failed to find wcs file \"%s\"\n", fn);
				continue;
			}
			logmsg("Reading WCS file \"%s\"\n", wcsfn);
			res = sip_read_header_file(wcsfn, &wcs);
			if (!res)
				logmsg("failed to parse SIP header from %s\n", wcsfn);
			free(wcsfn);
			if (!res)
				continue;
			W = wcs.wcstan.imagew;
			H = wcs.wcstan.imageh;
			logmsg("Image W,H %i, %i\n", W, H);

			if (strcmp("userdot", args->currentlayer) == 0) {
				double px, py;
				double ix, iy;
				double ra, dec;
				ix = W/2;
				iy = H/2;
				sip_pixelxy2radec(&wcs, ix, iy, &ra, &dec);
				px = ra2pixelf(ra, args);
				py = dec2pixelf(dec, args);
				cairo_move_to(cairo, px, py);
				cairo_arc(cairo, px, py, lw, 0, 2.0*M_PI);
				cairo_fill(cairo);
				continue;
			}

			cairo_set_line_width(cairo, lw);

			{
				// bottom, right, top, left, close.
				int offsetx[] = { 0, W, W, 0 };
				int offsety[] = { 0, 0, H, H };
				double stepx[] = { 1, 0, -1, 0 };
				double stepy[] = { 0, 1, 0, -1 };
				int side;
				double lastx=0, lasty=0;
				double lastra = 180.0;
				bool lastvalid = FALSE;
				double sx, sy;
				int nsides = 4;

				sx = W / (double)(nsteps-1);
				sy = H / (double)(nsteps-1);
				
				for (i=0; i<nsides; i++) {
					stepx[i] *= sx;
					stepy[i] *= sy;
				}

				for (side=0; side<nsides; side++) {
					for (i=0; i<nsteps; i++) {
 						double xin, yin;
						double xout, yout;
						double ra, dec;
						bool first = (!side && !i);
						bool wrapped;
						bool thisvalid;
						xin = offsetx[side] + i * stepx[side];
						yin = offsety[side] + i * stepy[side];
						sip_pixelxy2radec(&wcs, xin, yin, &ra, &dec);
						xout = ra2pixelf(ra, args);
						yout = dec2pixelf(dec, args);
						thisvalid = (yout > 0 && xout > 0 && yout < args->H && xout < args->W);
						wrapped = ((lastra < 90 && ra > 270) || 
								   (lastra > 270 && ra < 90));
						logmsg("image x,y %.1f, %.1f -> RA,Dec %.2f, %.2f -> plot x,y %.1f, %.1f.  Valid? %c; Wrapped? %c\n",
							   xin, yin, ra, dec, xout, yout, thisvalid ? 'T':'F', wrapped ? 'T':'F');
						if (wrapped)
							logmsg("Wrapped: lastra=%g, ra=%g, thisvalid=%i, lastvalid=%i, first=%i.\n",
								   lastra, ra, thisvalid, lastvalid, first);
						if (thisvalid && !lastvalid && !first && !wrapped)
							cairo_move_to(cairo, lastx, lasty);
						if (thisvalid)
							if (!wrapped)
								cairo_line_to(cairo, xout, yout);
						if (!thisvalid && lastvalid) {
							if (!wrapped)
								cairo_line_to(cairo, xout, yout);
						}
						if (wrapped)
							thisvalid = FALSE;
						lastra = ra;
						lastx = xout;
						lasty = yout;
						lastvalid = thisvalid;
					}
				}
			}
			//cairo_close_path(cairo);
			if (fill)
				cairo_fill(cairo);
			else
				cairo_stroke(cairo);

			if (args->dashbox > 0.0) {
				double ra, dec;
				double mx, my;
				double dm = 0.5 * args->dashbox;
				double dashes[] = {5, 5};
				// merc coordinate of field center:
				sip_pixelxy2radec(&wcs, 0.5 * W, 0.5 * H, &ra, &dec);
				mx = ra2merc(deg2rad(ra));
				my = dec2merc(deg2rad(dec));
				cairo_set_dash(cairo, dashes, sizeof(dashes)/sizeof(double), 0.0);
				draw_line_merc(mx-dm, my-dm, mx-dm, my+dm, cairo, args);
				draw_line_merc(mx-dm, my+dm, mx+dm, my+dm, cairo, args);
				draw_line_merc(mx+dm, my+dm, mx+dm, my-dm, cairo, args);
				draw_line_merc(mx+dm, my-dm, mx-dm, my-dm, cairo, args);
				cairo_stroke(cairo);

				if (args->zoomright) {
					cairo_set_line_width(cairo, lw/2.0);
					// draw lines from the left edge of the dashed box to the
					// right-hand edge of the image.
					cairo_set_dash(cairo, dashes, 0, 0.0);
					draw_line_merc(mx-dm, my-dm, args->xmercmax, args->ymercmin,
								   cairo, args);
					draw_line_merc(mx-dm, my+dm, args->xmercmax, args->ymercmax,
								   cairo, args);
					cairo_stroke(cairo);
				}
				if (args->zoomdown) {
					cairo_set_line_width(cairo, lw/2.0);
					cairo_set_dash(cairo, dashes, 0, 0.0);
					draw_line_merc(mx-dm, my+dm, args->xmercmin, args->ymercmin,
								   cairo, args);
					draw_line_merc(mx+dm, my+dm, args->xmercmax, args->ymercmin,
								   cairo, args);
					cairo_stroke(cairo);
				}
			}

		} else if (starts_with(arg, "bnsteps ")) {
			nsteps = get_int_arg(arg, nsteps);
			logmsg("Set nsteps = %i\n", nsteps);
		} else if (starts_with(arg, "bcolor ")) {
			get_double_args(arg, dargs);
			if (dl_size(dargs) != 3) {
				logmsg("argument 'bcolor' needs three doubles, got %i.\n", dl_size(dargs));
				return -1;
			}
			r = dl_get(dargs, 0);
			g = dl_get(dargs, 1);
			b = dl_get(dargs, 2);
			dl_remove_all(dargs);
			cairo_set_source_rgba(cairo, r, g, b, alpha);
		} else if (starts_with(arg, "balpha ")) {
			alpha = MAX(0, MIN(1, get_double_arg(arg, 1)));
			logmsg("Set alpha = %g\n", alpha);
			cairo_set_source_rgba(cairo, r, g, b, alpha);
		} else if (starts_with(arg, "bfill ")) {
			fill = (get_int_arg(arg, 0) == 1);
			logmsg("Set fill %s\n", (fill ? "on" : "off"));
		} else if (starts_with(arg, "blw ")) {
			lw = get_double_arg(arg, lw);
			logmsg("Set lw %g\n", lw);
		}
	}

	return 0;
}
