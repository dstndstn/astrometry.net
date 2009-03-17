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

	cairo_set_source_rgba(cairo, 0.5,0.8,0.8,0.3);

	for (i=0; i<sl_size(fns); i++) {
		char* fn;
        index_t* index;
        char* qidxfn;
        qidxfile* qidx;
		double* radec;
		int j, nstars;
		double crad = 3.0;
		int* starids;
		bool* starsinbounds;
		il* quadids;
		int nin;

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

		// FIXME - there could be a quad that partially appears in
		// this image but whose stars would not be detected by this
		// search.  Technically we should expand the radius by the
		// quad radius or so.

		startree_search_for(index->starkd, center, r2, NULL, &radec, &starids, &nstars);
		logmsg("found %i stars in the search radius\n", nstars);

		starsinbounds = malloc(sizeof(bool) * nstars);
		nin = 0;
		for (j=0; j<nstars; j++) {
			double px, py;
			px =  ra2pixelf(radec[2*j+0], args);
			py = dec2pixelf(radec[2*j+1], args);
            // cairo coords.
            px += 0.5;
            py += 0.5;
			starsinbounds[j] = in_image(px, py, args);
			if (starsinbounds[j])
				nin++;
		}
		logmsg("found %i stars inside the image bounds\n", nin);

		quadids = il_new(256);
		for (j=0; j<nstars; j++) {
			uint32_t* quads;
			int nquads;
			int k;
			if (!starsinbounds[j])
				continue;
			qidxfile_get_quads(qidx, starids[j], &quads, &nquads);
			for (k=0; k<nquads; k++)
				il_insert_unique_ascending(quadids, quads[k]);
		}
		logmsg("found %i quads involving stars inside the image bounds\n", il_size(quadids));

		for (j=0; j<il_size(quadids); j++) {
			int quadid;
			int qstarids[DQMAX];
			int DQ;
			int k;
			double starxy[DQMAX*2];
			double angles[DQMAX];
			double cx, cy;
			int perm[DQMAX];
			quadid = il_get(quadids, j);
			quadfile_get_stars(index->quads, quadid, qstarids);
			DQ = index_get_quad_dim(index);
			for (k=0; k<DQ; k++) {
				double ra, dec;
				double px, py;
				startree_get_radec(index->starkd, qstarids[k], &ra, &dec);
				px =  ra2pixelf(ra , args);
				py = dec2pixelf(dec, args);
				starxy[k*2 + 0] = px;
				starxy[k*2 + 1] = py;
			}
			cx = (starxy[0*2 + 0] + starxy[1*2 + 0]) / 2.0;
			cy = (starxy[0*2 + 1] + starxy[1*2 + 1]) / 2.0;
			for (k=0; k<DQ; k++)
				angles[k] = atan2(starxy[k*2 + 1] - cy, starxy[k*2 + 0] - cx);
			permutation_init(perm, DQ);
			permuted_sort(angles, sizeof(double), compare_doubles_asc, perm, DQ);
			for (k=0; k<DQ; k++) {
				double px, py;
				px = starxy[perm[k]*2+0];
				py = starxy[perm[k]*2+1];
				if (k == 0)
					cairo_move_to(cairo, px, py);
				else
					cairo_line_to(cairo, px, py);
			}
			cairo_close_path(cairo);
			cairo_fill(cairo);
			for (k=0; k<DQ; k++) {
				double px, py;
				px = starxy[perm[k]*2+0];
				py = starxy[perm[k]*2+1];
				if (k == 0)
					cairo_move_to(cairo, px, py);
				else
					cairo_line_to(cairo, px, py);
			}
			cairo_close_path(cairo);
			cairo_stroke(cairo);
		}

		free(starids);
		free(starsinbounds);

        qidxfile_close(qidx);
        index_close(index);
	}

	sl_free2(fns);
	return 0;
}

