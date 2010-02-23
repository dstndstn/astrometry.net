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
#include "qidxfile.h"
#include "permutedsort.h"

static void logmsg(char* format, ...) {
	va_list args;
	va_start(args, format);
	fprintf(stderr, "render_quads: ");
	vfprintf(stderr, format, args);
	va_end(args);
}

void quad_radec_to_xy(render_args_t* args, const double* radecs,
					  double* xys, int DQ) {
	int k;
	double angles[DQMAX];
	double cx, cy;
	int perm[DQMAX];
	for (k=0; k<DQ; k++) {
		xys[2*k + 0] =  ra2pixelf(radecs[2*k + 0], args);
		xys[2*k + 1] = dec2pixelf(radecs[2*k + 1], args);
	}
	cx = (xys[0*2 + 0] + xys[1*2 + 0]) / 2.0;
	cy = (xys[0*2 + 1] + xys[1*2 + 1]) / 2.0;
	for (k=0; k<DQ; k++)
		angles[k] = atan2(xys[k*2 + 1] - cy, xys[k*2 + 0] - cx);
	permutation_init(perm, DQ);
	permuted_sort(angles, sizeof(double), compare_doubles_asc, perm, DQ);
	permutation_apply(perm, DQ, xys, xys, 2*sizeof(double));
}

int render_quads(cairo_t* cairo, render_args_t* args) {
	sl* fns;
	int i;
	double center[3];
	double r2;
	double p1[3], p2[3];
	double edge_rgba[4];
	double face_rgba[4];
	bool edge_set = FALSE;
	bool face_set = FALSE;
	double alpha = 0.3;

	fns = sl_new(256);
	get_string_args_of_type(args, "index ", fns);

	cairo_set_line_join(cairo, CAIRO_LINE_JOIN_ROUND);

	if (!get_first_rgba_arg_of_type(args, "quadedgergba ", edge_rgba))
		edge_set = TRUE;
	if (!get_first_rgba_arg_of_type(args, "quadfacergba ", face_rgba))
		face_set = TRUE;

    logmsg("got %i index files.\n", sl_size(fns));

	//// FIXME -- this fails for all-sky (eg ramin=0, ramax=360) because it underestimates!
	radecdeg2xyzarr(args->ramin, args->decmin, p1);
	radecdeg2xyzarr(args->ramax, args->decmax, p2);
	star_midpoint(center, p1, p2);
	r2 = distsq(p1, center, 3);

	for (i=0; i<sl_size(fns); i++) {
		char* fn;
        index_t* index;
        char* qidxfn;
        qidxfile* qidx;
		double* radec;
		int j, nstars;
		int* starids;
		il* quadids;
        double quadr2;

		fn = sl_get(fns, i);
        index = index_load(fn, 0, NULL);
		if (!index) {
			logmsg("failed to open index from file \"%s\"\n", fn);
			continue;
		}

        qidxfn = index_get_qidx_filename(index->indexname);
		qidx = qidxfile_open(qidxfn);
		if (!qidx) {
			logmsg("Failed to open qidxfile \"%s\".\n", qidxfn);
            exit(-1);            
		}

        quadr2 = arcsec2distsq(index->index_scale_upper);
		startree_search_for(index->starkd, center, r2+quadr2, NULL, &radec, &starids, &nstars);
		logmsg("found %i stars in the search radius\n", nstars);

		quadids = il_new(256);
		for (j=0; j<nstars; j++) {
			uint32_t* quads;
			int nquads;
			int k;
			qidxfile_get_quads(qidx, starids[j], &quads, &nquads);
			for (k=0; k<nquads; k++)
				il_insert_unique_ascending(quadids, quads[k]);
		}
		logmsg("found %i quads involving stars inside the image bounds\n", il_size(quadids));

		for (j=0; j<il_size(quadids); j++) {
			int quadid;
			unsigned int qstarids[DQMAX];
			int DQ;
			int k;
			double starxy[DQMAX*2];
			double quadradec[DQMAX*2];
            double r,g,b;
			quadid = il_get(quadids, j);
			quadfile_get_stars(index->quads, quadid, qstarids);
			DQ = index_get_quad_dim(index);
			for (k=0; k<DQ; k++) {
				startree_get_radec(index->starkd, qstarids[k],
								   quadradec+2*k, quadradec+2*k+1);
			}
			quad_radec_to_xy(args, quadradec, starxy, DQ);

			if (face_set)
				cairo_set_source_rgba(cairo, face_rgba[0], face_rgba[1], face_rgba[2], face_rgba[3]);
			else {
				srand(quadid);
				r = ((rand() % 128) + 127)/255.0;
				g = ((rand() % 128) + 127)/255.0;
				b = ((rand() % 128) + 127)/255.0;
				cairo_set_source_rgba(cairo, r,g,b,alpha);
			}
			cairoutils_draw_path(cairo, starxy, DQ);
			cairo_close_path(cairo);
			cairo_fill(cairo);

			if (edge_set)
				cairo_set_source_rgba(cairo, edge_rgba[0], edge_rgba[1], edge_rgba[2], edge_rgba[3]);
			else {
				srand(quadid);
				r = ((rand() % 128) + 127)/255.0;
				g = ((rand() % 128) + 127)/255.0;
				b = ((rand() % 128) + 127)/255.0;
				cairo_set_source_rgba(cairo, r,g,b,alpha);
			}
			cairoutils_draw_path(cairo, starxy, DQ);
			cairo_close_path(cairo);
			cairo_stroke(cairo);
		}

		free(starids);
        qidxfile_close(qidx);
        index_close(index);
	}

	sl_free2(fns);
	return 0;
}

