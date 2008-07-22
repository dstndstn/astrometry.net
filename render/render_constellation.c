/*
   This file is part of the Astrometry.net suite.
   Copyright 2007 Keir Mierle and Dustin Lang.

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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <sys/types.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cairo.h>

#include "tilerender.h"
#include "render_constellation.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "cairoutils.h"
#include "constellations.h"

int render_constellation(unsigned char* img, render_args_t* args) {
	cairo_t* cairo;
	cairo_surface_t* target;
	double lw = args->linewidth;
	int i;
	int c;
    int N;

	srand(0);

	fprintf(stderr, "render_constellation: Starting.\n");

	target = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, args->W, args->H, args->W*4);
	cairo = cairo_create(target);
	cairo_set_line_width(cairo, lw);
	cairo_set_line_join(cairo, CAIRO_LINE_JOIN_BEVEL);
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);
	cairo_set_source_rgb(cairo, 1.0, 1.0, 1.0);

    N = constellations_n();
	for (c=0; c<N; c++) {
		const char* longname;
        il* lines;
		il* uniqstars;
		double cmass[3];
		double ra,dec;
		double px,py;
		unsigned char r,g,b;
        int Nunique;

        uniqstars = constellations_get_unique_stars(c);
        Nunique = il_size(uniqstars);

		r = (rand() % 128) + 127;
		g = (rand() % 128) + 127;
		b = (rand() % 128) + 127;
        cairo_set_source_rgba(cairo, r/255.0,g/255.0,b/255.0,0.8);

        // find center of mass (of the in-bounds stars)
        cmass[0] = cmass[1] = cmass[2] = 0.0;
        for (i=0; i<Nunique; i++) {
            double xyz[3];
            int star = il_get(uniqstars, i);
            constellations_get_star_radec(star, &ra, &dec);
            radecdeg2xyzarr(ra, dec, xyz);
            cmass[0] += xyz[0];
            cmass[1] += xyz[1];
            cmass[2] += xyz[2];
        }
        cmass[0] /= Nunique;
        cmass[1] /= Nunique;
        cmass[2] /= Nunique;
        xyzarr2radecdeg(cmass, &ra, &dec);
		px = ra2pixel(ra, args);
		py = dec2pixel(dec, args);
        longname = constellations_get_longname(c);
        assert(longname);

		cairo_select_font_face(cairo, "helvetica", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
		cairo_set_font_size(cairo, 14.0);
        cairo_move_to(cairo, px, py);
        cairo_show_text(cairo, longname);
        cairo_stroke(cairo);

        il_free(uniqstars);

        // Draw the lines.
        cairo_set_line_width(cairo, lw);
        lines = constellations_get_lines(c);
        for (i=0; i<il_size(lines)/2; i++) {
            int star1, star2;
            double ra1, dec1, ra2, dec2;
			int SEGS=20;

            star1 = il_get(lines, i*2+0);
            star2 = il_get(lines, i*2+1);
            constellations_get_star_radec(star1, &ra1, &dec1);
            constellations_get_star_radec(star2, &ra2, &dec2);
            draw_segmented_line(ra1, dec1, ra2, dec2, SEGS, cairo, args);
            cairo_stroke(cairo);
        }
        il_free(lines);
	}
	fprintf(stderr, "render_constellations: Read %i constellations.\n", c);

    cairoutils_argb32_to_rgba(img, args->W, args->H);

	cairo_surface_destroy(target);
	cairo_destroy(cairo);

	return 0;
}
