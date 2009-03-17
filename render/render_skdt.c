#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_skdt.h"
#include "starutil.h"
#include "mathutil.h"
#include "mercrender.h"
#include "cairoutils.h"
#include "starkd.h"

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_skdt: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

int render_skdt(cairo_t* cairo, render_args_t* args) {
	sl* fns;
	int i;
	double center[3];
	double r2;
	double p1[3], p2[3];

	fns = sl_new(256);
	get_string_args_of_type(args, "skdt ", fns);

    logmsg("got %i skdt files.\n", sl_size(fns));

	radecdeg2xyzarr(args->ramin, args->decmin, p1);
	radecdeg2xyzarr(args->ramax, args->decmax, p2);
	star_midpoint(center, p1, p2);
	r2 = distsq(p1, center, 3);

	cairo_set_source_rgba(cairo, 0,1,0,1);

	for (i=0; i<sl_size(fns); i++) {
		char* fn;
		startree_t* skdt;
		double* radec;
		int j, nstars;
		double crad = 3.0;

		fn = sl_get(fns, i);
		skdt = startree_open(fn);
		if (!skdt) {
			logmsg("failed to open star kdtree from file \"%s\"\n", fn);
			continue;
		}
		logmsg("reading star kdtree \"%s\"\n", fn);

		startree_search(skdt, center, r2, NULL, &radec, &nstars);

		for (j=0; j<nstars; j++) {
			double px, py;
			px =  ra2pixelf(radec[2*j+0], args);
			py = dec2pixelf(radec[2*j+1], args);
            // cairo coords.
            px += 0.5;
            py += 0.5;
			if (!in_image_margin(px, py, crad, args))
				continue;
            cairo_move_to(cairo, px+crad, py);
			cairo_arc(cairo, px, py, crad, 0.0, 2.0*M_PI);
		}
        cairo_stroke(cairo);

		free(radec);
		startree_close(skdt);
	}

	sl_free2(fns);
	return 0;
}

