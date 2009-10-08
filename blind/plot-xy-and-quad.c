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

/*
 Hackery for dstn's thesis - cairo pdf rendering of "plotxy" and "plotquad"
 over a background image.
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/param.h>

#include <cairo.h>
#include <cairo-pdf.h>

#include "xylist.h"
#include "matchfile.h"
#include "permutedsort.h"
#include "boilerplate.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"

int main(int argc, char *args[]) {
	int W, H;
	unsigned char* img = NULL;
	cairo_t* cairo;
	cairo_surface_t* target;
	xylist_t* xyls;
	starxy_t* xy;
	int Nxy;
	int Nstars = 0;
	int ext = 1;
	double xoff = 0.0, yoff = 0.0;
    float br=0.0, bg=0.0, bb=0.0;
    float r=1.0, g=1.0, b=1.0;
	double rad = 5.0;
	double lw = 1.0;
    int marker;
	int i;
	int nquads;
	int dimquads = 4;
	dl* coords;

	/*
	 #	| plotxy -i m88.xy -I - -x 1 -y 1 -b white -C black -N 100 -r 10 -w 2 -P \
	 #	| plotquad -I - -C black -m m88-${I2}.match -P \
	 */

	// HACK -- hard-coded args.
	char* bgimgfn = "m88-bw.jpg";
	char* xylsfn = "m88.xy";
	char* matchfn = "m88-9702.match";

	log_init(LOG_MSG);
	log_to(stderr);
	errors_log_to(stderr);

	Nstars = 100;
	xoff = yoff = 1.0;
	r = g = b = 0.0;
	br = bg = bb = 1.0;
	rad = 10;
	lw = 2;
	marker = CAIROUTIL_MARKER_CIRCLE;

	W = H = -1;
	coords = dl_new(16);

	// read bg image to get size.
	img = cairoutils_read_jpeg(bgimgfn, &W, &H);
	if (!img) {
		ERROR("Failed to read background image \"%s\"", bgimgfn);
		exit(-1);
	}
	logmsg("Read \"%s\": %ix%i pixels\n", bgimgfn, W, H);

	xyls = xylist_open(xylsfn);
	if (!xyls) {
		ERROR("Failed to read xylist \"%s\"", xylsfn);
		exit(-1);
	}

	// create output buffer.
	target = cairo_pdf_surface_create_for_stream(cairoutils_file_write_func, stdout, W, H);
	if (!target) {
		ERROR("Failed to create cairo surface for PDF");
		exit(-1);
	}
	cairo = cairo_create(target);
	cairo_set_line_width(cairo, lw);
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);

	cairoutils_surface_status_errors(target);
	cairoutils_cairo_status_errors(cairo);

	// render image.
	{
		cairo_surface_t* thissurf;
		cairo_pattern_t* pat;
		cairoutils_rgba_to_argb32(img, W, H);
		thissurf = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, W, H, W*4);
		pat = cairo_pattern_create_for_surface(thissurf);
		cairo_set_source(cairo, pat);
		cairo_paint(cairo);
		cairo_pattern_destroy(pat);
		cairo_surface_destroy(thissurf);
	}
	free(img);

	// render xylist.
    // we don't care about FLUX and BACKGROUND columns.
    xylist_set_include_flux(xyls, FALSE);
    xylist_set_include_background(xyls, FALSE);
	// Find number of entries in xylist.
    xy = xylist_read_field_num(xyls, ext, NULL);
    if (!xy) {
		ERROR("Failed to read FITS extension %i from file %s", ext, xylsfn);
		exit(-1);
	}
    Nxy = starxy_n(xy);
	logmsg("Xylist contains %i stars\n", Nxy);
	// If N is specified, apply it as a max.
    if (Nstars) {
		if (Nstars < Nxy)
			logmsg("Keeping %i stars.\n", Nstars);
        Nxy = MIN(Nxy, Nstars);
	}

	cairo_set_source_rgb(cairo, r, g, b);

	// render background markers.
	cairo_save(cairo);
	cairo_set_line_width(cairo, lw+2.0);
	cairo_set_source_rgba(cairo, br, bg, bb, 0.75);
	for (i=0; i<Nxy; i++) {
		double x = starxy_getx(xy, i) + 0.5 - xoff;
		double y = starxy_gety(xy, i) + 0.5 - yoff;
		cairoutils_draw_marker(cairo, marker, x, y, rad);
		cairo_stroke(cairo);
	}
	cairo_restore(cairo);

	// Draw markers.
	for (i=0; i<Nxy; i++) {
		double x = starxy_getx(xy, i) + 0.5 - xoff;
		double y = starxy_gety(xy, i) + 0.5 - yoff;
        cairoutils_draw_marker(cairo, marker, x, y, rad);
		cairo_stroke(cairo);
	}

    starxy_free(xy);
	xylist_close(xyls);

	// Plot quad.
	{
        matchfile* mf = matchfile_open(matchfn);
        MatchObj* mo;
        if (!mf) {
            ERROR("Failed to open matchfile \"%s\"", matchfn);
            exit(-1);
        }
        while (1) {
            mo = matchfile_read_match(mf);
            if (!mo)
                break;
            for (i=0; i<2*dimquads; i++) {
                dl_append(coords, mo->quadpix[i]);
            }
        }
	}
	nquads = dl_size(coords) / (2*dimquads);

	lw = 4;
	cairo_set_line_width(cairo, lw);
	cairo_set_line_join(cairo, CAIRO_LINE_JOIN_BEVEL);
	for (i=0; i<nquads; i++) {
		int j;
		double theta[dimquads];
		int perm[dimquads];
		double cx, cy;

		// Make the quad convex so Sam's eyes don't bleed.
		cx = cy = 0.0;
		for (j=0; j<dimquads; j++) {
			cx += dl_get(coords, i*(2*dimquads) + j*2);
			cy += dl_get(coords, i*(2*dimquads) + j*2 + 1);
		}
		cx /= dimquads;
		cy /= dimquads;
		for (j=0; j<dimquads; j++) {
			theta[j] = atan2(dl_get(coords, i*(2*dimquads) + j*2 + 1)-cy,
							 dl_get(coords, i*(2*dimquads) + j*2 + 0)-cx);
		}
		permutation_init(perm, dimquads);
		permuted_sort(theta, sizeof(double), compare_doubles_asc, perm, dimquads);
		for (j=0; j<dimquads; j++) {
			((j==0) ? cairo_move_to : cairo_line_to)
                (cairo,
                 dl_get(coords, i*(2*dimquads) + perm[j]*2),
                 dl_get(coords, i*(2*dimquads) + perm[j]*2 + 1));
		}
		cairo_close_path(cairo);
		cairo_stroke(cairo);
	}


	// do output & clean up.
	cairo_surface_flush(target);
	cairo_surface_finish(target);
	cairoutils_surface_status_errors(target);
	cairoutils_cairo_status_errors(cairo);

	cairo_surface_destroy(target);
	cairo_destroy(cairo);

	return 0;
}
