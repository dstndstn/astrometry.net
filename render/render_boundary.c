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

#include <cairo.h>

#include "tilerender.h"
#include "render_boundary.h"
#include "sip_qfits.h"
#include "cairoutils.h"
#include "ioutils.h"

const char* wcs_dirs[] = {
	"/home/gmaps/ontheweb-data",
	"/home/gmaps/test/web-data",
	"/home/gmaps/apod-solves",
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
	sl* wcsfiles = NULL;
	dl* colorlist = NULL;
	double r, g, b;

	r = b = 0;
    g = 1.0;

	logmsg("Starting.\n");

    colorlist = dl_new(256);
    get_double_args_of_type(args, "color ", colorlist);
    if (!dl_size(colorlist)) {
        dl_free(colorlist);
        colorlist = NULL;
    }

    wcsfiles = sl_new(256);
    get_string_args_of_type(args, "bwcsfn ", wcsfiles);
	if (!sl_size(wcsfiles)) {
		logmsg("No WCS files specified.\n");
		return -1;
	}
    if (colorlist && dl_size(colorlist) != sl_size(wcsfiles)*3) {
		logmsg("Color list has %i entries, expected 3 * WCS files (%i) = %i\n",
               dl_size(colorlist), sl_size(wcsfiles), sl_size(wcsfiles)*3);
		return -1;
    }

	cairo_set_line_width(cairo, lw);
	cairo_set_line_join(cairo, CAIRO_LINE_JOIN_ROUND);
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);
	cairo_set_source_rgb(cairo, r, g, b);

    for (I=0; I<sl_size(wcsfiles); I++) {
		char* fn;
        char* wcsfn;
        sip_t* res;
		sip_t wcs;
		int W, H;

        fn = sl_get(wcsfiles, I);
        wcsfn = find_file_in_dirs(wcs_dirs, sizeof(wcs_dirs)/sizeof(char*), fn, TRUE);
        if (!wcsfn) {
            logmsg("failed to find wcs file \"%s\"\n", fn);
            continue;
		}

        res = sip_read_header_file(wcsfn, &wcs);
        if (!res)
            logmsg("failed to parse SIP header from %s\n", wcsfn);
        free(wcsfn);
        if (!res)
            continue;
        W = wcs.wcstan.imagew;
        H = wcs.wcstan.imageh;

        if (colorlist) {
            double rr,gg,bb;
            rr = dl_get(colorlist, I*3+0);
            gg = dl_get(colorlist, I*3+1);
            bb = dl_get(colorlist, I*3+2);
            cairo_set_source_rgb(cairo, rr, gg, bb);
        }

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

        } else {
			// bottom, right, top, left, close.
			int offsetx[] = { 0, W, W, 0, 0 };
			int offsety[] = { 0, 0, H, H, 0 };
			int stepx[] = { 10, 0, -10, 0, 0 };
			int stepy[] = { 0, 10, 0, -10, 0 };
			int Nsteps[] = { W/10, H/10, W/10, H/10, 1 };
			int side;
			double lastx=0, lasty=0;
			double lastra = 180.0;
			bool lastvalid = FALSE;

			for (side=0; side<5; side++) {
				for (i=0; i<Nsteps[side]; i++) {
					int xin, yin;
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

					if (wrapped) {
						logmsg("Wrapped: lastra=%g, ra=%g, thisvalid=%i, lastvalid=%i, first=%i.\n",
							   lastra, ra, thisvalid, lastvalid, first);
					}

					if (thisvalid && !lastvalid && !first && !wrapped)
						cairo_move_to(cairo, lastx, lasty);
					if (thisvalid)
						if (!wrapped)
							cairo_line_to(cairo, xout, yout);
					if (!thisvalid && lastvalid) {
						if (!wrapped)
							cairo_line_to(cairo, xout, yout);
						cairo_stroke(cairo);
					}

					if (wrapped)
						thisvalid = FALSE;
					lastra = ra;
					lastx = xout;
					lasty = yout;
					lastvalid = thisvalid;
                }
            }
			cairo_stroke(cairo);
		}

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
	}

    if (colorlist)
        dl_free(colorlist);

	sl_free2(wcsfiles);

	return 0;
}
