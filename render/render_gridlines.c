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
					 const char* txt, render_args_t* args, double* bgrgba) {
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
	cairo_set_source_rgba(cairo, bgrgba[0], bgrgba[1], bgrgba[2], bgrgba[3]);
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

int render_gridlines(cairo_t* cairo, render_args_t* args) {
	double rastep, decstep;
	int ind;
	double steps[] = { 1, 20.0, 10.0, 6.0, 4.0, 2.5, 1.0, 30./60.0,
					   15.0/60.0, 10.0/60.0, 5.0/60.0, 2./60.0 };
	double ra, dec;
	double ralabelstep=0, declabelstep=0;
	double gridrgba[] = { 0.8,0.8,0.8,0.8 };
	double textrgba[] = { 0.8,0.8,0.8,0.8 };
	double textbgrgba[] = { 0,0,0,0.8 };

	ind = MAX(1, args->zoomlevel);
	ind = MIN(ind, sizeof(steps)/sizeof(double)-1);
	rastep = decstep = steps[ind];
	rastep = get_double_arg_of_type(args, "gridrastep ", rastep);
	decstep = get_double_arg_of_type(args, "griddecstep ", decstep);
	logmsg("Grid step: RA %g, Dec %g.\n", rastep, decstep);
	cairo_set_line_width(cairo, 1.0);
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);

	if (args->gridlabel) {
		ralabelstep = 2. * rastep;
		declabelstep = 2. * decstep;
		ralabelstep = get_double_arg_of_type(args, "gridlabelrastep ", decstep);
		declabelstep = get_double_arg_of_type(args, "gridlabeldecstep ", declabelstep);
		logmsg("Grid label step: RA %g, Dec %g.\n", ralabelstep, declabelstep);
	}

	get_first_rgba_arg_of_type(args, "grid_rgba ", gridrgba);
	cairo_set_source_rgba(cairo, gridrgba[0], gridrgba[1], gridrgba[2], gridrgba[3]);

	get_first_rgba_arg_of_type(args, "grid_textrgba ", textrgba);
	get_first_rgba_arg_of_type(args, "grid_textbgrgba ", textbgrgba);
	if (args->gridlabel) {
		int fontsize = 18;
		cairo_select_font_face(cairo, "DejaVu Sans Mono Book", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
		fontsize = get_int_arg(get_first_arg_of_type(args, "grid_textfontsize "), fontsize);
		logmsg("Grid label font size: %i\n", fontsize);
		cairo_set_font_size(cairo, fontsize);
	}

	for (ra = rastep * floor(args->ramin / rastep);
		 ra <= rastep * ceil(args->ramax / rastep);
		 ra += rastep) {
		float x = ra2pixelf(ra, args);
		if (!in_image((int)round(x), 0, args))
			continue;
		// draw the grid line on the nearest pixel... cairo pixel centers are at 0.5
		x = 0.5 + round(x-0.5);
		cairo_move_to(cairo, x, 0);
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
		char buf[32];
		cairo_set_source_rgba(cairo, textrgba[0], textrgba[1], textrgba[2], textrgba[3]);
		for (ra = ralabelstep * floor(args->ramin / ralabelstep);
			 ra <= ralabelstep * ceil(args->ramax / ralabelstep);
			 ra += ralabelstep) {
			float x = ra2pixelf(ra, args);
			float y = args->H;
			if (!in_image((int)round(x+0.5), 0, args))
				continue;
			pretty_label(ra, buf);
			logverb("Adding label ra=\"%s\"\n", buf);
			add_text(cairo, x, y, buf, args, textbgrgba);
		}
		for (dec = declabelstep * floor(args->decmin / declabelstep);
			 dec <= declabelstep * ceil(args->decmax / declabelstep);
			 dec += declabelstep) {
			float y = dec2pixelf(dec, args);
			float x = 0;
			// yep, it can wrap around :)
			if ((dec > 90) || (dec < -90))
				continue;
			if (!in_image(0, (int)round(y+0.5), args))
				continue;
			pretty_label(dec, buf);
			logverb("Adding label dec=\"%s\"\n", buf);
			add_text(cairo, x, y, buf, args, textbgrgba);
		}
		cairo_stroke(cairo);
	}
	return 0;
}
