/*
   This file is part of the Astrometry.net suite.
   Copyright 2000 Dustin Lang.

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
#include <stdarg.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_images.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"

int render_cairo(cairo_t* cairo, render_args_t* args) {
	sl* cmds;
	int i;
	logmsg("starting.\n");

	cmds = sl_new(256);
	get_string_args_of_type(args, "cairo", cmds);
	logmsg("Found %i cairo commands\n", sl_size(cmds));
	for (i=0; i<sl_size(cmds); i++) {
		sl* words;
		char* cmd;
		words = sl_split(NULL, sl_get(cmds, i), " ");
		// remove blank at front.
		sl_remove(words, 0);
		if (sl_size(words) == 0) {
			sl_free2(words);
			continue;
		}
		cmd = sl_get(words, 0);
		if (streq(cmd, "moveto") ||
			streq(cmd, "lineto")) {
			double ra,dec;
			float x, y;
			if (sl_size(words) < 3) {
				logmsg("need <ra> <dec> for moveto or lineto\n");
				return -1;
			}
			ra = atof(sl_get(words, 1));
			dec = atof(sl_get(words, 2));
			x = ra2pixelf(ra, args);
			y = dec2pixelf(dec, args);
			if (streq(cmd, "moveto")) {
				cairo_move_to(cairo, x, y);
				logverb("move to %g,%g\n", x, y);
			} else if (streq(cmd, "lineto")) {
				cairo_line_to(cairo, x, y);
				logverb("line to %g,%g\n", x, y);
			}
		} else if (streq(cmd, "color")) {
			float r,g,b,a;
			if (!((sl_size(words) == 4) || (sl_size(words) == 5))) {
				logmsg("need <r> <g> <b> [<a>] for color\n");
				return -1;
			}
			r = atof(sl_get(words, 1));
			g = atof(sl_get(words, 2));
			b = atof(sl_get(words, 3));
			if (sl_size(words) == 5) {
				a = atof(sl_get(words, 4));
			} else {
				a = 1.0;
			}
			logverb("set color %g,%g,%g,%g\n", r,g,b,a);
			cairo_set_source_rgba(cairo, r, g, b, a);
		} else if (streq(cmd, "stroke")) {
			cairo_stroke(cairo);
		} else {
			logmsg("didn't understand command \"%s\"\n", cmd);
		}
		sl_free2(words);
	}
	sl_free2(cmds);
	cairo_stroke(cairo);
	return 0;
}

