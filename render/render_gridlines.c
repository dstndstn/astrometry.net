#include <stdio.h>
#include <math.h>
#include <sys/param.h>
#include <string.h>

#include <cairo.h>

#include "tilerender.h"
#include "render_gridlines.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"

static void add_text(cairo_t* cairo, double x, double y,
					 const char* txt, render_args_t* args) {
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
	if (r > args->W)
		ex = -(r - args->W);
	if (b > args->H)
		ey = -(b - args->H);
	x += ex;
	l += ex;
	r += ex;
	y += ey;
	t += ey;
	b += ey;

	cairo_save(cairo);
	// blank out underneath the text...
	cairo_set_source_rgba(cairo, 0, 0, 0, 0.8);
	cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
	cairo_move_to(cairo, l, t);
	cairo_line_to(cairo, l, b);
	cairo_line_to(cairo, r, b);
	cairo_line_to(cairo, r, t);
	cairo_close_path(cairo);
	cairo_fill(cairo);
	cairo_stroke(cairo);
	cairo_restore(cairo);

	cairo_move_to(cairo, x, y);
	cairo_show_text(cairo, txt);
}

int render_gridlines(cairo_t* c2, render_args_t* args) {
	double rastep, decstep;
	int ind;
	double steps[] = { 1, 20.0, 10.0, 6.0, 4.0, 2.5, 1.0, 30./60.0,
					   15.0/60.0, 10.0/60.0, 5.0/60.0, 2./60.0 };
	double ra, dec;
	cairo_t* cairo;
	cairo_surface_t* mask;
	double ralabelstep, declabelstep;

	ind = MAX(1, args->zoomlevel);
	ind = MIN(ind, sizeof(steps)/sizeof(double)-1);
	rastep = decstep = steps[ind];
	rastep = get_double_arg_of_type(args, "gridrastep ", rastep);
	decstep = get_double_arg_of_type(args, "griddecstep ", decstep);
	logmsg("Grid step: RA %g, Dec %g.\n", rastep, decstep);

	if (args->gridlabel) {
		ralabelstep = 2. * rastep;
		declabelstep = 2. * decstep;
		ralabelstep = get_double_arg_of_type(args, "gridlabelrastep ", decstep);
		declabelstep = get_double_arg_of_type(args, "gridlabeldecstep ", declabelstep);
		logmsg("Grid label step: RA %g, Dec %g.\n", ralabelstep, declabelstep);
	}

	/*
	 In order to properly do the transparency and text, we render onto a
	 mask image, then squish paint through this mask onto the given image.
	 */
	mask = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, args->W, args->H);
   	cairo = cairo_create(mask);
	cairo_set_line_width(cairo, 1.0);
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);
	cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.7);

	for (ra = rastep * floor(args->ramin / rastep);
		 ra <= rastep * ceil(args->ramax / rastep);
		 ra += rastep) {
		float x = ra2pixelf(ra, args);
		if (!in_image((int)round(x), 0, args))
			continue;
		// draw the grid line on the nearest pixel... cairo pixel centers are at 0.5
		x = 0.5 + round(x-0.5);
		cairo_move_to(cairo, x, 0);
		//cairo_line_to(cairo, x, args->H - y0);
		cairo_line_to(cairo, x, args->H);
	}
	for (dec = decstep * floor(args->decmin / decstep);
		 dec <= decstep * ceil(args->decmax / decstep);
		 dec += decstep) {
		float y = dec2pixelf(dec, args);
		if (!in_image(0, (int)round(y), args))
			continue;
		y = 0.5 + round(y-0.5);
		cairo_move_to(cairo, 0, y);
		cairo_line_to(cairo, args->W, y);
	}
	cairo_stroke(cairo);
	
	if (args->gridlabel) {
		cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 8.0);
		cairo_select_font_face(cairo, "DejaVu Sans Mono Book", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
		cairo_set_font_size(cairo, 18);
		for (ra = ralabelstep * floor(args->ramin / ralabelstep);
			 ra <= ralabelstep * ceil(args->ramax / ralabelstep);
			 ra += ralabelstep) {
			char buf[32];
			float x = ra2pixelf(ra, args);
			float y = args->H;
			if (!in_image((int)round(x+0.5), 0, args))
				continue;
			sprintf(buf, "%.2f", ra);
			// Trim off ".00"
			if (ends_with(buf, ".00"))
				buf[strlen(buf) - 3] = '\0';
			add_text(cairo, x, y, buf, args);
		}
		for (dec = declabelstep * floor(args->decmin / declabelstep);
			 dec <= declabelstep * ceil(args->decmax / declabelstep);
			 dec += declabelstep) {
			char buf[32];
			float y = dec2pixelf(dec, args);
			float x = 0;
			// yep, it can wrap around :)
			if ((dec > 90) || (dec < -90))
				continue;
			if (!in_image(0, (int)round(y+0.5), args))
				continue;
			sprintf(buf, "%.2f", dec);
			// Trim off ".00"
			if (ends_with(buf, ".00"))
				buf[strlen(buf) - 3] = '\0';
			add_text(cairo, x, y, buf, args);
		}
	}

	// squish paint through mask
	cairo_set_source_surface(c2, mask, 0, 0);
	cairo_mask_surface(c2, mask, 0, 0);

	cairo_surface_destroy(mask);
	cairo_destroy(cairo);

	return 0;
}
