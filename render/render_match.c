#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/param.h>

#include "tilerender.h"
#include "render_match.h"
#include "render_quads.h"
#include "starutil.h"
#include "mathutil.h"
#include "mercrender.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "matchfile.h"
#include "errors.h"
#include "permutedsort.h"

/*
 static void logmsg(char* format, ...) {
 va_list args;
 va_start(args, format);
 fprintf(stderr, "render_match: ");
 vfprintf(stderr, format, args);
 va_end(args);
 }
 */

int render_match(cairo_t* cairo, render_args_t* args) {
	int i, I;
	double edge_rgba[] = { 0,1,0,1 };
	double face_rgba[] = { 1,1,1,0 };

	for (I=0; I<sl_size(args->arglist); I++) {
		char* arg = sl_get(args->arglist, I);
		if (starts_with(arg, "matchfn ")) {
			matchfile* mf;
			char* fn;
			fn = arg + strlen("matchfn ");
			mf = matchfile_open(fn);
			if (!mf) {
				ERROR("Failed to open match file \"%s\"", fn);
				return -1;
			}
			while (1) {
				double radec[DQMAX*2];
				double xy[DQMAX*2];
				MatchObj* mo = matchfile_read_match(mf);
				if (!mo)
					break;
				for (i=0; i<mo->dimquads; i++)
					xyzarr2radecdegarr(mo->quadxyz + 3*i, radec + 2*i);
				quad_radec_to_xy(args, radec, xy, mo->dimquads);

				cairoutils_draw_path(cairo, xy, mo->dimquads);
				cairo_close_path(cairo);
				cairo_set_source_rgba(cairo, face_rgba[0], face_rgba[1], face_rgba[2], face_rgba[3]);
				cairo_fill(cairo);

				cairoutils_draw_path(cairo, xy, mo->dimquads);
				cairo_close_path(cairo);
				cairo_set_source_rgba(cairo, edge_rgba[0], edge_rgba[1], edge_rgba[2], edge_rgba[3]);
				cairo_stroke(cairo);
			}
		} else if (starts_with(arg, "match_edge_rgba ")) {
			if (parse_rgba_arg(arg, edge_rgba)) {
				return -1;
			}
		} else if (starts_with(arg, "match_face_rgba ")) {
			if (parse_rgba_arg(arg, face_rgba)) {
				return -1;
			}
		}
	}
	return 0;
}

