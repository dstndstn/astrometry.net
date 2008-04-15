#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_rdls.h"
#include "rdlist.h"
#include "starutil.h"
#include "mathutil.h"
#include "mercrender.h"
#include "cairoutils.h"

char* rdls_dirs[] = {
    "/home/gmaps/gmaps-rdls/",
    "/home/gmaps/ontheweb-data/",
};

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_rdls: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

int render_rdls(unsigned char* img, render_args_t* args)
{
    int i, j, Nstars, Nib=0;
	int r;
	cairo_t* cairo;
	cairo_surface_t* target;

	// draw black outline?
	bool outline = TRUE;

	target = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32,
												 args->W, args->H, args->W*4);
	cairo = cairo_create(target);
	cairo_set_line_join(cairo, CAIRO_LINE_JOIN_BEVEL);
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);

	for (r=0; r<pl_size(args->rdlsfns); r++) {
		char* rdlsfn = pl_get(args->rdlsfns, r);
		char* color = pl_get(args->rdlscolors, r);
		int fieldnum = il_get(args->fieldnums, r);
		int maxstars = il_get(args->Nstars, r);
		//double lw = dl_get(args->rdlslws, r);
		double lw = 2.0;
		double rad = 3.0;
		char style = 'o';
		double r, g, b;
        rdlist_t* rdls;
        rd_t* rd;

		cairo_set_line_width(cairo, lw);
		r = g = b = 1.0;

		for (j=0; color && j<strlen(color); j++) {
			if (parse_color(color[j], &r, &g, &b) == 0)
				continue;
			switch (color[j]) {
			case 'p':
			case '+': // plus-sign
				style = '+';
				break;
			case 'h':
			case '#': // box
				style = '#';
				break;
			case 'o': // circle
				style = 'o';
				break;
			}
		}
		cairo_set_source_rgb(cairo, r, g, b);

		
		/* Search in the rdls paths */
		for (i=0; i<sizeof(rdls_dirs)/sizeof(char*); i++) {
			char fn[256];
			snprintf(fn, sizeof(fn), "%s/%s", rdls_dirs[i], rdlsfn);
			logmsg("Trying file: %s\n", fn);
			rdls = rdlist_open(fn);
			if (rdls)
				break;
		}
		if (!rdls) {
			logmsg("Failed to open RDLS file \"%s\".\n", rdlsfn);
			return -1;
		}

		if (fieldnum < 1)
			fieldnum = 1;

        rd = rdlist_read_field_num(rdls, fieldnum, NULL);
        if (!rd) {
			logmsg("Failed to read RDLS file. \"%s\"\n", rdlsfn);
			return -1;
		}
        Nstars = rd_n(rd);
        if (maxstars)
            Nstars = MIN(Nstars, maxstars);
		logmsg("Got %i stars.\n", Nstars);

		Nib = 0;
		for (i=0; i<Nstars; i++) {
			double px, py;

			px =  ra2pixelf(rd_getra (rd, i), args);
			py = dec2pixelf(rd_getdec(rd, i), args);

			if (!in_image_margin(px, py, rad, args))
				continue;

			// cairo has funny ideas about pixel coordinates...
			px += 0.5;
			py += 0.5;

			switch (style) {
			case 'o':
				if (outline) {
					cairo_set_line_width(cairo, lw + 2.0);
					cairo_set_source_rgb(cairo, 0, 0, 0);
					cairo_arc(cairo, px, py, rad, 0.0, 2.0*M_PI);
					cairo_stroke(cairo);
					cairo_set_line_width(cairo, lw);
					cairo_set_source_rgb(cairo, r, g, b);
				}
				cairo_arc(cairo, px, py, rad, 0.0, 2.0*M_PI);

				break;

			case '+':
				cairo_move_to(cairo, px-rad, py);
				cairo_line_to(cairo, px+rad, py);
				cairo_move_to(cairo, px, py-rad);
				cairo_line_to(cairo, px, py+rad);
				break;

			case '#':
				cairo_move_to(cairo, px-rad, py-rad);
				cairo_line_to(cairo, px-rad, py+rad);
				cairo_line_to(cairo, px+rad, py+rad);
				cairo_line_to(cairo, px+rad, py-rad);
				cairo_line_to(cairo, px-rad, py-rad);
				break;
			}

			cairo_stroke(cairo);

			Nib++;
		}

        rd_free(rd);
        rdlist_close(rdls);

    }
    logmsg("%i stars inside image bounds.\n", Nib);

    cairoutils_argb32_to_rgba(img, args->W, args->H);
	
	cairo_surface_destroy(target);
	cairo_destroy(cairo);

    return 0;
}
