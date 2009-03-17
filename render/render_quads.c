#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_quads.h"
#include "starutil.h"
#include "mathutil.h"
#include "mercrender.h"
#include "cairoutils.h"
#include "index.h"
#include "starkd.h"

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_quads: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

int render_quads(cairo_t* cairo, render_args_t* args) {
	sl* fns;
	int i;
	double center[3];
	double r2;
	double p1[3], p2[3];

	fns = sl_new(256);
	get_string_args_of_type(args, "index ", fns);

    logmsg("got %i index files.\n", sl_size(fns));

	radecdeg2xyzarr(args->ramin, args->decmin, p1);
	radecdeg2xyzarr(args->ramax, args->decmax, p2);
	star_midpoint(center, p1, p2);
	r2 = distsq(p1, center, 3);

	cairo_set_source_rgba(cairo, 0,1,0,1);

	for (i=0; i<sl_size(fns); i++) {
		char* fn;
        index_t* index;
        char* qidxfn;
        qidxfile* qidx;
		double* radec;
		int j, nstars;
		double crad = 3.0;

		fn = sl_get(fns, i);
        index = index_load(fn, 0);
		if (!index) {
			logmsg("failed to open index from file \"%s\"\n", fn);
			continue;
		}

        qidxfn = index_get_qidx_filename(index->meta.indexname);
		qidx = qidxfile_open(qidxfn);
		if (!qidx) {
			logmsg("Failed to open qidxfile \"%s\".\n", qidxfn);
            exit(-1);            
		}


        qidxfile_close(qidx);
        index_close(index);
	}

	sl_free2(fns);
	return 0;
}

