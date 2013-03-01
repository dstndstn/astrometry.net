/*
 This file is part of the Astrometry.net suite.
 Copyright 2007-2008 Dustin Lang, Keir Mierle and Sam Roweis.
 Copyright 2009, 2010, 2012 Dustin Lang.

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

#include <sys/param.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <cairo.h>

#include "plotannotations.h"
#include "hd.h"
#include "ngc2000.h"
#include "brightstars.h"
#include "cairoutils.h"
#include "sip-utils.h"
#include "starutil.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "sip-utils.h"
#include "mathutil.h"
#include "constellations.h"

DEFINE_PLOTTER(annotations);

struct target {
	double ra;
	double dec;
	char* name;
};
typedef struct target target_t;

plotann_t* plot_annotations_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "annotations");
}

static void plot_targets(cairo_t* cairo, plot_args_t* pargs, plotann_t* ann) {
	int i;
	double cra, cdec;
	plotstuff_get_radec_center_and_radius(pargs, &cra, &cdec, NULL);
	
	for (i=0; i<bl_size(ann->targets); i++) {
		target_t* tar = bl_access(ann->targets, i);
		double px,py;
		double cx,cy;
		double dx,dy, r;
		double ex,ey;
		double ly, ry, tx, bx;
		double distdeg;
		anbool okquadrant;
		char* txt;

		logverb("Target: \"%s\" at (%g,%g)\n", tar->name, tar->ra, tar->dec);
		okquadrant = plotstuff_radec2xy(pargs, tar->ra, tar->dec, &px, &py);

		if (okquadrant &&
			px >= 0 && px < pargs->W && py >= 0 && py < pargs->H) {
			// inside the image!
			logverb("Target \"%s\" is inside the image, at pixel (%g,%g)\n", tar->name, px, py);
			plotstuff_stack_marker(pargs, px, py);
			plotstuff_stack_text(pargs, cairo, tar->name, px, py);
			continue;
		}

		// outside the image: find intersection point.
		cx = pargs->W / 2.0;
		cy = pargs->H / 2.0;
		if (okquadrant) {
			logverb("Target \"%s\" is outside the image, at pixel (%g,%g)\n", tar->name, px, py);
			dx = px - cx;
			dy = py - cy;
		} else {
			double cxyz[3];
			double txyz[3];
			double vec[3];
			int j;
			double ra,dec;
			logverb("Target \"%s\" is way outside the image.\n", tar->name);
			// fallback.
			radecdeg2xyzarr(cra, cdec, cxyz);
			radecdeg2xyzarr(tar->ra, tar->dec, txyz);
			for (j=0; j<3; j++)
				vec[j] = cxyz[j] + 0.1 * txyz[j];
			normalize_3(vec);
			xyzarr2radecdeg(vec, &ra, &dec);
			okquadrant = plotstuff_radec2xy(pargs, ra, dec, &px, &py);
			assert(okquadrant);
			dx = px - cx;
			dy = py - cy;
			if ((dx*dx + dy*dy) < (cx*cx + cy*cy)) {
				double scale = 3.0 * sqrt(cx*cx + cy*cy) / sqrt(dx*dx + dy*dy);
				dx *= scale;
				dy *= scale;
			}
		}

		ly = (-(pargs->W/2.0) / dx) * dy + cy;
		ry = ( (pargs->W/2.0) / dx) * dy + cy;
		bx = (-(pargs->H/2.0) / dy) * dx + cx;
		tx = ( (pargs->H/2.0) / dy) * dx + cx;
		logverb("ly %g, ry %g, bx %g, tx %g\n", ly, ry, bx, tx);
		if (px < cx && ly >= 0 && ly < pargs->H) {
			ex = 0.0;
			ey = ly;
		} else if (px >= cx && ry >= 0 && ry < pargs->H) {
			ex = pargs->W - 1;
			ey = ry;
		} else if (py < cy && bx >= 0 && bx < pargs->W) {
			ex = bx;
			ey = 0;
		} else if (py >= cy && tx >= 0 && tx < pargs->W) {
			ex = tx;
			ey = pargs->H - 1;
		} else {
			logverb("None of the edges are in bounds: px,py=(%g,%g); ly=%g, ry=%g, bx=%g, tx=%g\n", px,py,ly,ry,bx,tx);
			continue;
		}
		dx = ex - cx;
		dy = ey - cy;
		r = sqrt(dx*dx + dy*dy);

		px = (r-100.0) / r * dx + cx;
		py = (r-100.0) / r * dy + cy;

		plotstuff_stack_arrow(pargs, px, py, ex, ey);
		logverb("Arrow from (%g,%g) to (%g,%g)\n", px, py, ex, ey);
		distdeg = deg_between_radecdeg(cra, cdec, tar->ra, tar->dec);
		asprintf(&txt, "%s: %.1f deg", tar->name, distdeg);
		plotstuff_stack_text(pargs, cairo, txt, px, py);
	}
}

static void plot_constellations(cairo_t* cairo, plot_args_t* pargs, plotann_t* ann) {
	int i, N;
	double ra,dec,radius;
	double xyzf[3];
	// Find the field center and radius
	anwcs_get_radec_center_and_radius(pargs->wcs, &ra, &dec, &radius);
	logverb("Plotting constellations: field center %g,%g, radius %g\n",
			ra, dec, radius);
	radecdeg2xyzarr(ra, dec, xyzf);
	radius = deg2dist(radius);

	N = constellations_n();
	for (i=0; i<N; i++) {
		int j, k;
		// Find the approximate center and radius of this constellation
		// and see if it overlaps with the field.
		il* stars = constellations_get_unique_stars(i);
		double xyzj[3];
		double xyzc[3];
		double maxr2 = 0;
		dl* rds;
		xyzc[0] = xyzc[1] = xyzc[2] = 0.0;
		for (j=0; j<il_size(stars); j++) {
			constellations_get_star_radec(il_get(stars, j), &ra, &dec);
			radecdeg2xyzarr(ra, dec, xyzj);
			for (k=0; k<3; k++)
				xyzc[k] += xyzj[k];
		}
		normalize_3(xyzc);
		for (j=0; j<il_size(stars); j++) {
			constellations_get_star_radec(il_get(stars, j), &ra, &dec);
			maxr2 = MAX(maxr2, distsq(xyzc, xyzj, 3));
		}
		il_free(stars);
		maxr2 = square(sqrt(maxr2) + radius);
		if (distsq(xyzf, xyzc, 3) > maxr2) {
			xyzarr2radecdeg(xyzc, &ra, &dec);
			logverb("Constellation %s (center %g,%g, radius %g) out of bounds\n",
					constellations_get_shortname(i), ra, dec,
					dist2deg(sqrt(maxr2) - radius));
			logverb("  dist from field center to constellation center is %g deg\n",
					distsq2deg(distsq(xyzf, xyzc, 3)));
			logverb("  max radius: %g\n", distsq2deg(maxr2));
			continue;
		}
		// Phew, plot it.
		if (ann->constellation_lines) {
			rds = constellations_get_lines_radec(i);
			logverb("Constellation %s: plotting %i lines\n",
					constellations_get_shortname(i), dl_size(rds)/4);
			for (j=0; j<dl_size(rds)/4; j++) {
				double r1,d1,r2,d2;
				double r3,d3,r4,d4;
				r1 = dl_get(rds, j*4+0);
				d1 = dl_get(rds, j*4+1);
				r2 = dl_get(rds, j*4+2);
				d2 = dl_get(rds, j*4+3);
				if (anwcs_find_discontinuity(pargs->wcs, r1, d1, r2, d2,
											 &r3, &d3, &r4, &d4)) {
					logverb("Discontinuous: %g,%g -- %g,%g\n", r1, d1, r2, d2);
					logverb("  %g,%g == %g,%g\n", r3,d3, r4,d4);
					plotstuff_move_to_radec(pargs, r1, d1);
					plotstuff_line_to_radec(pargs, r3, d3);
					plotstuff_move_to_radec(pargs, r4, d4);
					plotstuff_line_to_radec(pargs, r2, d2);
				} else {
					plotstuff_move_to_radec(pargs, r1, d1);
					plotstuff_line_to_radec(pargs, r2, d2);
				}
				plotstuff_stroke(pargs);
			}
			dl_free(rds);
		}

		if (ann->constellation_labels ||
			ann->constellation_markers) {
			// Put the label at the center of mass of the stars that are in-bounds
			int Nin = 0;
			stars = constellations_get_unique_stars(i);
			xyzc[0] = xyzc[1] = xyzc[2] = 0.0;
			logverb("Labeling %s: %i stars\n", constellations_get_shortname(i),
					il_size(stars));
			for (j=0; j<il_size(stars); j++) {
				constellations_get_star_radec(il_get(stars, j), &ra, &dec);
				if (!anwcs_radec_is_inside_image(pargs->wcs, ra, dec))
					continue;
				if (ann->constellation_markers) {
					plotstuff_marker_radec(pargs, ra, dec);
				}
				radecdeg2xyzarr(ra, dec, xyzj);
				for (k=0; k<3; k++)
					xyzc[k] += xyzj[k];
				Nin++;
			}
			logverb("  %i stars in-bounds\n", Nin);
			if (ann->constellation_labels && Nin) {
				const char* label;
				normalize_3(xyzc);
				xyzarr2radecdeg(xyzc, &ra, &dec);
				if (ann->constellation_labels_long)
					label = constellations_get_longname(i);
				else
					label = constellations_get_shortname(i);
				plotstuff_text_radec(pargs, ra, dec, label);
			}
			il_free(stars);
		}
	}
}

static void plot_brightstars(cairo_t* cairo, plot_args_t* pargs, plotann_t* ann) {
	int i, N;

	N = bright_stars_n();
	for (i=0; i<N; i++) {
		double px, py;
		char* label;
		const brightstar_t* bs = bright_stars_get(i);
		if (!plotstuff_radec2xy(pargs, bs->ra, bs->dec, &px, &py))
			continue;
		logverb("Bright star %s/%s at RA,Dec (%g,%g) -> xy (%g, %g)\n", bs->name, bs->common_name, bs->ra, bs->dec, px, py);
		if (px < 1 || py < 1 || px > pargs->W || py > pargs->H)
			continue;
		// skip unnamed
		if (!strlen(bs->name) && !strlen(bs->common_name))
			continue;

		label = (strlen(bs->common_name) ? bs->common_name : bs->name);
		plotstuff_stack_marker(pargs, px, py);
		plotstuff_stack_text(pargs, cairo, label, px, py);
	}
}

int plot_annotations_set_hd_catalog(plotann_t* ann, const char* hdfn) {
	if (ann->hd_catalog)
		free(ann->hd_catalog);
	ann->hd_catalog = strdup(hdfn);
	return 0;
}

static void plot_hd(cairo_t* cairo, plot_args_t* pargs, plotann_t* ann) {
	int i, N;
	hd_catalog_t* hdcat = NULL;
	double ra,dec,rad;
	bl* hdlist = NULL;

	if (!ann->hd_catalog)
		return;
	hdcat = henry_draper_open(ann->hd_catalog);
	if (!hdcat) {
		ERROR("Failed to open Henry Draper catalog file \"%s\"", ann->hd_catalog);
		return;
	}
	if (plotstuff_get_radec_center_and_radius(pargs, &ra, &dec, &rad)) {
		ERROR("Failed to get RA,Dec,radius from plotstuff");
		return;
	}
	hdlist = henry_draper_get(hdcat, ra, dec, deg2arcsec(rad));
	logverb("Got %i Henry Draper stars\n", bl_size(hdlist));
	
	N = bl_size(hdlist);
	for (i=0; i<N; i++) {
		hd_entry_t* entry = bl_access(hdlist, i);
		double px, py;
		char label[16];
		if (!plotstuff_radec2xy(pargs, entry->ra, entry->dec, &px, &py))
			continue;
		if (px < 1 || py < 1 || px > pargs->W || py > pargs->H)
			continue;
		logverb("HD %i at RA,Dec (%g,%g) -> xy (%g, %g)\n", entry->hd, entry->ra, entry->dec, px, py);

		sprintf(label, "HD %i", entry->hd);
		plotstuff_stack_marker(pargs, px, py);
		plotstuff_stack_text(pargs, cairo, label, px, py);
	}
	bl_free(hdlist);
	henry_draper_close(hdcat);
}

static void plot_ngc(cairo_t* cairo, plot_args_t* pargs, plotann_t* ann) {
	double imscale;
	double imsize;
	int i, N;

	// arcsec/pixel
	imscale = plotstuff_pixel_scale(pargs);
	// arcmin
	imsize = imscale * MIN(pargs->W, pargs->H) / 60.0;

	N = ngc_num_entries();
	logverb("Checking %i NGC/IC objects.\n", N);

	for (i=0; i<N; i++) {
		ngc_entry* ngc;
		char* names;
		double pixrad;
		double px, py;
		double r;

		ngc = ngc_get_entry_accurate(i);
		if (!ngc)
			break;
		if (ngc->size < imsize * ann->ngc_fraction) {
			// FIXME -- just plot an X-mark with label.
			debug("%s %i: size %g arcmin < limit of %g\n",
				  (ngc->is_ngc ? "NGC":"IC"), ngc->id, ngc->size, imsize*ann->ngc_fraction);
			continue;
		}

		if (!plotstuff_radec2xy(pargs, ngc->ra, ngc->dec, &px, &py)) {
			debug("%s %i: RA,Dec (%.1f,%.1f) is >90 deg away.\n",
				  (ngc->is_ngc ? "NGC":"IC"), ngc->id, ngc->ra, ngc->dec);
			continue;
		}

		pixrad = 0.5 * ngc->size * 60.0 / imscale;
		if (px < -pixrad || py < -pixrad || px > pargs->W + pixrad || py > pargs->H + pixrad) {
			debug("%s %i: RA,Dec (%.1f,%.1f), pix (%.1f,%.1f) is out-of-bounds\n",
				  (ngc->is_ngc ? "NGC":"IC"), ngc->id, ngc->ra, ngc->dec, px, py);
			continue;
		}

		names = ngc_get_name_list(ngc, " / ");
		printf("%s\n", names);

		logverb("%s %i: RA,Dec (%.1f,%.1f), size %g arcmin, pix (%.1f,%.1f), radius %g\n",
				(ngc->is_ngc ? "NGC":"IC"), ngc->id, ngc->ra, ngc->dec, ngc->size, px, py, pixrad);
		debug("size: %f arcsec, pix radius: %f pixels\n", ngc->size, pixrad);
		// save old marker size...
		r = pargs->markersize;
		pargs->markersize = pixrad;
		plotstuff_stack_marker(pargs, px, py);
		plotstuff_stack_text(pargs, cairo, names, px, py);
		free(names);
		// revert old marker size...
		pargs->markersize = r;

		/*
		 if (json) {
		 char* namelist = sl_implode(names, "\", \"");
		 sl_appendf(json,
		 "{ \"type\"   : \"ngc\", "
		 "  \"names\"  : [ \"%s\" ], "
		 "  \"pixelx\" : %g, "
		 "  \"pixely\" : %g, "
		 "  \"radius\" : %g }"
		 , namelist, px, py, pixsize/2.0);
		 free(namelist);
		 }
		 */
	}
}

void* plot_annotations_init(plot_args_t* args) {
	plotann_t* ann = calloc(1, sizeof(plotann_t));
	ann->ngc_fraction = 0.02;
	ann->targets = bl_new(4, sizeof(target_t));
	ann->NGC = TRUE;
	ann->bright = TRUE;
	ann->constellation_lines = TRUE;
	return ann;
}

int plot_annotations_plot(const char* cmd, cairo_t* cairo,
							 plot_args_t* pargs, void* baton) {
	plotann_t* ann = (plotann_t*)baton;

	// Set fonts, etc, before calling plotting routines
	plotstuff_builtin_apply(cairo, pargs);

	if (ann->NGC)
		plot_ngc(cairo, pargs, ann);

	if (ann->bright)
		plot_brightstars(cairo, pargs, ann);

	if (ann->HD)
		plot_hd(cairo, pargs, ann);

	if (ann->constellations)
		plot_constellations(cairo, pargs, ann);

	if (bl_size(ann->targets))
		plot_targets(cairo, pargs, ann);

	return plotstuff_plot_stack(pargs, cairo);
}

int plot_annotations_command(const char* cmd, const char* cmdargs,
							 plot_args_t* pargs, void* baton) {
	plotann_t* ann = (plotann_t*)baton;
	if (streq(cmd, "annotations_no_ngc")) {
		ann->NGC = FALSE;
	} else if (streq(cmd, "annotations_no_bright")) {
		ann->bright = FALSE;
	} else if (streq(cmd, "annotations_ngc_size")) {
		ann->ngc_fraction = atof(cmdargs);
	} else if (streq(cmd, "annotations_target")) {
		sl* args = sl_split(NULL, cmdargs, " ");
		double ra, dec;
		char* name;
		if (sl_size(args) != 3) {
			ERROR("Need RA,Dec,name");
			return -1;
		}
		ra = atof(sl_get(args, 0));
		dec = atof(sl_get(args, 1));
		name = sl_get(args, 2);
		plot_annotations_add_target(ann, ra, dec, name);
	} else if (streq(cmd, "annotations_targetname")) {
		const char* name = cmdargs;
		return plot_annotations_add_named_target(ann, name);
	} else {
		ERROR("Unknown command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

int plot_annotations_add_named_target(plotann_t* ann, const char* name) {
	target_t tar;
	ngc_entry* e = ngc_get_entry_named(name);
	if (!e) {
		ERROR("Failed to find target named \"%s\"", name);
		return -1;
	}
	tar.name = ngc_get_name_list(e, " / ");
	tar.ra = e->ra;
	tar.dec = e->dec;
	logmsg("Found %s: RA,Dec (%g,%g)\n", tar.name, tar.ra, tar.dec);
	bl_append(ann->targets, &tar);
	return 0;
}

void plot_annotations_add_target(plotann_t* ann, double ra, double dec,
								 const char* name) {
	target_t tar;
	memset(&tar, 0, sizeof(target_t));
	tar.ra = ra;
	tar.dec = dec;
	tar.name = strdup(name);
	logmsg("Added target \"%s\" at (%g,%g)\n", tar.name, tar.ra, tar.dec);
	bl_append(ann->targets, &tar);
}

void plot_annotations_free(plot_args_t* args, void* baton) {
	plotann_t* ann = (plotann_t*)baton;
	free(ann->hd_catalog);
	free(ann);
}



