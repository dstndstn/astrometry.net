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

const char* rdls_dirs[] = {
    "/home/gmaps/gmaps-rdls",
    "/home/gmaps/ontheweb-data",
    ".",
};

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_rdls: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

static int draw_markers(int Nstars, rd_t* rd, render_args_t* args, cairo_t* cairo,
						double rad, char style) {
	int Nib = 0;
	int i;
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

		case 'x':
			cairo_move_to(cairo, px-rad, py-rad);
			cairo_line_to(cairo, px+rad, py+rad);
			cairo_move_to(cairo, px+rad, py-rad);
			cairo_line_to(cairo, px-rad, py+rad);
			break;

		}
		cairo_stroke(cairo);
		Nib++;
	}
	return Nib;
}

int render_rdls(cairo_t* cairo, render_args_t* args) {
    int j, Nstars, Nib=0;
	int r;
	double outrgba[4];

	// draw outline behind markers?
	bool outline = FALSE;

	logmsg("Rendering %i rdls files\n", pl_size(args->rdlsfns));

	for (r=0; r<pl_size(args->rdlsfns); r++) {
		char* rdlsfn = pl_get(args->rdlsfns, r);
		char* color = pl_get(args->rdlscolors, r);
		int fieldnum = il_get(args->fieldnums, r);
		int maxstars = il_get(args->Nstars, r);
		//double lw = dl_get(args->rdlslws, r);
		double lw = 2.0;
		double out_lw;
		double rad;
		double out_rad;
		char style = 'o';
		double r, g, b;
        rdlist_t* rdls;
        rd_t* rd;
		char* path;

		r = 1.0;
        g = b = 0.0;

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
			case 'x': // x
				style = 'x';
				break;
			case 'o': // circle
				style = 'o';
				break;
			}
		}
		
		/* Search in the rdls paths */
		path = find_file_in_dirs(rdls_dirs, sizeof(rdls_dirs)/sizeof(char*), rdlsfn, TRUE);
		if (!path) {
			logmsg("Failed to find RDLS file \"%s\"\n", rdlsfn);
			return -1;
		}
		rdls = rdlist_open(path);
		free(path);
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

		rad = get_first_double_arg_of_type(args, "rdls_rad ", 3.0);
		out_rad = get_first_double_arg_of_type(args, "rdls_outrad ", rad);
		lw = get_first_double_arg_of_type(args, "rdls_lw ", 2.0);
		out_lw = get_first_double_arg_of_type(args, "rdls_outlw ", lw + 2);
		outline = (get_first_rgba_arg_of_type(args, "rdls_outrgba ", outrgba) == 0);

		if (outline) {
			cairo_set_line_width(cairo, out_lw);
			cairo_set_source_rgba(cairo, outrgba[0], outrgba[1], outrgba[2], outrgba[3]);
			draw_markers(Nstars, rd, args, cairo, out_rad, style);
		}

		cairo_set_line_width(cairo, lw);
		cairo_set_source_rgb(cairo, r, g, b);
		Nib = draw_markers(Nstars, rd, args, cairo, rad, style);

        rd_free(rd);
        rdlist_close(rdls);

    }
    logmsg("%i stars inside image bounds.\n", Nib);
    return 0;
}
