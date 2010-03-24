/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "plotindex.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "starutil.h"
#include "index.h"

const plotter_t plotter_index = {
	.name = "index",
	.init = plot_index_init,
	.command = plot_index_command,
	.doplot = plot_index_plot,
	.free = plot_index_free
};

void* plot_index_init(plot_args_t* plotargs) {
	plotindex_t* args = calloc(1, sizeof(plotindex_t));
	args->indexes = pl_new(16);
	args->stars = TRUE;
	args->quads = TRUE;
	return args;
}

int plot_index_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotindex_t* args = (plotindex_t*)baton;
	int i;
	double ra, dec, radius;
	double xyz[3];
	double r2;

	if (plotstuff_get_radec_center_and_radius(pargs, &ra, &dec, &radius)) {
		ERROR("Failed to get RA,Dec center and radius");
		return -1;
	}
	radecdeg2xyzarr(ra, dec, xyz);
	r2 = deg2distsq(radius);

	for (i=0; i<pl_size(args->indexes); i++) {
		index_t* index = pl_get(args->indexes, i);
		int j, N;
		int k, DQ;
		double* radecs;
		double px,py;

		if (args->stars) {
			// plot stars
			startree_search_for(index->starkd, xyz, r2, NULL, &radecs, NULL, &N);
			logmsg("Found %i stars in range of index %s\n", N, index->indexname);
			for (j=0; j<N; j++) {
				if (!plotstuff_radec2xy(pargs, radecs[j*2], radecs[j*2+1], &px, &py)) {
					ERROR("Failed to convert RA,Dec %g,%g to pixels\n", radecs[j*2], radecs[j*2+1]);
					continue;
				}
				cairoutils_draw_marker(cairo, pargs->marker, px, py, pargs->markersize);
				cairo_stroke(cairo);
			}
			free(radecs);
		}
		if (args->quads) {
			// plot quads
			N = index_nquads(index);
			DQ = index_get_quad_dim(index);
			// HACK -- could use quadidx if the index is much bigger than the plot area...
			for (j=0; j<N; j++) {
				unsigned int stars[DQMAX];
				double ra, dec;
				quadfile_get_stars(index->quads, j, stars);
				for (k=0; k<DQ; k++) {
					startree_get_radec(index->starkd, stars[k], &ra, &dec);
					if (!plotstuff_radec2xy(pargs, ra, dec, &px, &py)) {
						ERROR("Failed to convert RA,Dec %g,%g to pixels for quad %i\n", ra, dec, j);
						continue;
					}
					if (k == 0) {
						cairo_move_to(cairo, px, py);
					} else {
						cairo_line_to(cairo, px, py);
					}
				}
				cairo_close_path(cairo);
				cairo_stroke(cairo);
			}
		}
	}
	return 0;
}

int plot_index_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotindex_t* args = (plotindex_t*)baton;
	if (streq(cmd, "index_file")) {
		const char* fn = cmdargs;
		index_t* index = index_load(fn, 0, NULL);
		if (!index) {
			ERROR("Failed to open index \"%s\"", fn);
			return -1;
		}
		pl_append(args->indexes, index);
	} else if (streq(cmd, "index_draw_stars")) {
		args->stars = atoi(cmdargs);
	} else if (streq(cmd, "index_draw_quads")) {
		args->quads = atoi(cmdargs);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_index_free(plot_args_t* plotargs, void* baton) {
	plotindex_t* args = (plotindex_t*)baton;
	int i;
	for (i=0; i<pl_size(args->indexes); i++) {
		index_t* index = pl_get(args->indexes, i);
		index_free(index);
	}
	pl_free(args->indexes);
	free(args);
}

