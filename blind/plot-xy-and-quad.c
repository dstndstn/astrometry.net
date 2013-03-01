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
#include <cairo-ps.h>

#include "xylist.h"
#include "matchfile.h"
#include "permutedsort.h"
#include "boilerplate.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hvj:x:m:W:H:s:E";

static void printHelp(char* progname) {
	fprintf(stderr, "\nUsage: %s [options] > output.pdf\n"
			"   -j <jpeg filename>\n"
			"   -x <xylist filename>\n"
			"   -m <match filename>\n"
			"  [-W <output width (points)>]\n"
			"  [-H <output height (points)>]\n"
			"  [-s <xylist-sale-factor>]\n"
			"  [-E]: EPS output\n"
			"\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *args[]) {
	int argchar;
	char* progname = args[0];

	int outW=0, outH=0;
	int W, H;
	unsigned char* img = NULL;
	cairo_t* cairo;
	cairo_surface_t* target;
	xylist_t* xyls;
	starxy_t* xy;
	int Nxy;
	int Nstars;
	int ext = 1;
	double xoff, yoff;
	double scalexy = 1.0;
    float br, bg, bb;
    float r, g, b;
	double rad;
	double lw;
    int marker;
	int i;
	int nquads;
	int dimquads = 4;
	dl* coords;
	int loglvl = LOG_MSG;
	double sx, sy;
	anbool eps = FALSE;

	/*
	 #	| plotxy -i m88.xy -I - -x 1 -y 1 -b white -C black -N 100 -r 10 -w 2 -P \
	 #	| plotquad -I - -C black -m m88-${I2}.match -P \
	 */

	char* bgimgfn = NULL;
	char* xylsfn = NULL;
	char* matchfn = NULL;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
		case 'E':
			eps = TRUE;
			break;
		case 'j':
			bgimgfn = optarg;
			break;
		case 'x':
			xylsfn = optarg;
			break;
		case 'm':
			matchfn = optarg;
			break;
		case 'W':
			outW = atoi(optarg);
			break;
		case 'H':
			outH = atoi(optarg);
			break;
		case 's':
			scalexy = atof(optarg);
			break;
		case 'h':
			printHelp(progname);
            exit(0);
		case 'v':
			loglvl++;
			break;
		}

	log_init(LOG_MSG);
	log_to(stderr);
	errors_log_to(stderr);

	if (!(bgimgfn && xylsfn && matchfn)) {
		ERROR("Must specify background jpeg, xyls, and match filenames.\n");
		printHelp(progname);
		exit(-1);
	}

	Nstars = 100;
	xoff = yoff = 1.0;
	r = g = b = 0.0;
	br = bg = bb = 1.0;
	rad = 6;
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

	if (!outW)
		outW = W;
	if (!outH)
		outH = H;

	logmsg("Background image aspect ratio %g; output file aspect ratio %g.\n", W/(float)H, outW/(float)outH);

	// create output buffer.
	if (eps) {
		target = cairo_ps_surface_create_for_stream(cairoutils_file_write_func, stdout, outW, outH);
		cairo_ps_surface_set_eps(target, TRUE);
	} else {
		target = cairo_pdf_surface_create_for_stream(cairoutils_file_write_func, stdout, outW, outH);
	}
	if (!target) {
		ERROR("Failed to create cairo surface");
		exit(-1);
	}
	cairo = cairo_create(target);

	sx = outW/(float)W;
	sy = outH/(float)H;

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

		logmsg("Scaling image by factors %g, %g\n", sx, sy);
		cairo_save(cairo);
		cairo_scale(cairo, sx, sy);
		cairo_set_source(cairo, pat);
		cairo_paint(cairo);
		cairo_restore(cairo);

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
	//cairo_set_source_rgba(cairo, br, bg, bb, 0.75);
	cairo_set_source_rgba(cairo, br, bg, bb, 1.0);
	for (i=0; i<Nxy; i++) {
		double x = (starxy_getx(xy, i) - xoff) * sx * scalexy + 0.5;
		double y = (starxy_gety(xy, i) - yoff) * sy * scalexy + 0.5;
		cairoutils_draw_marker(cairo, marker, x, y, rad);
		cairo_stroke(cairo);
	}
	cairo_restore(cairo);

	// Draw markers.
	for (i=0; i<Nxy; i++) {
		double x = (starxy_getx(xy, i) - xoff) * sx * scalexy + 0.5;
		double y = (starxy_gety(xy, i) - yoff) * sy * scalexy + 0.5;
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
                 (dl_get(coords, i*(2*dimquads) + perm[j]*2)
				  - xoff) * sx * scalexy + 0.5,
                 (dl_get(coords, i*(2*dimquads) + perm[j]*2 + 1)
				  - yoff) * sy * scalexy + 0.5);
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
