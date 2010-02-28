/*
 This file is part of the Astrometry.net suite.
 Copyright 2007-2008 Dustin Lang, Keir Mierle and Sam Roweis.
 Copyright 2009 Dustin Lang.

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

#include <cairo.h>

#include "plotannotations.h"
#include "ngc2000.h"
#include "cairoutils.h"
#include "sip-utils.h"
#include "starutil.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

const plotter_t plotter_annotations = {
	.name = "annotations",
	.init = plot_annotations_init,
	.command = plot_annotations_command,
	.doplot = plot_annotations_plot,
	.free = plot_annotations_free
};

enum cmdtype {
	CIRCLE,
	TEXT,
	LINE,
	RECTANGLE,
	ARROW,
	MARKER,
};
typedef enum cmdtype cmdtype;

struct cairocmd {
	cmdtype type;
	int layer; // ?
	double x, y;
	float rgba[4];
	//char* color;
	// CIRCLE
	double radius;
	// TEXT
	char* text;
	// LINE / RECTANGLE / ARROW
	double x2, y2;
	// MARKER
	int marker;
	double markersize;
};
typedef struct cairocmd cairocmd_t;


struct target {
	double ra;
	double dec;
	char* name;
};
typedef struct target target_t;

struct annotation_args {
	bool NGC;
	bool constellations;
	bool bright;
	bool HD;
	float ngc_fraction;
	bl* cairocmds;
	float fontsize;
	char* fontname;
	float bg_rgba[4];
	bl* targets;
};
typedef struct annotation_args ann_t;



static void add_text(plot_args_t* pargs, ann_t* ann, cairo_t* cairo,
                     const char* txt, double px, double py) {
    cairo_text_extents_t textents;
    double l,r,t,b;
    double margin = 2.0;
    int dx, dy;
	cairocmd_t cmd;
	memset(&cmd, 0, sizeof(cmd));

    cairo_text_extents(cairo, txt, &textents);
    l = px + textents.x_bearing;
    r = l + textents.width + textents.x_bearing;
    t = py + textents.y_bearing;
    b = t + textents.height;
    l -= margin;
    t -= margin;
    r += margin + 1;
    b += margin + 1;

    // move text away from the edges of the image.
    if (l < 0) {
        px += -l;
        l = 0;
    }
    if (t < 0) {
        py += -t;
        t = 0;
    }
    if (r > pargs->W) {
        px -= (r - pargs->W);
        r = pargs->W;
    }
    if (b > pargs->H) {
        py -= (b - pargs->H);
        b = pargs->H;
    }

	cmd.type = TEXT;
	cmd.layer = 2;
	memcpy(cmd.rgba, ann->bg_rgba, sizeof(cmd.rgba));
    for (dy=-1; dy<=1; dy++) {
        for (dx=-1; dx<=1; dx++) {
			cmd.text = strdup(txt);
			cmd.x = px + dx;
			cmd.y = py + dy;
			bl_append(ann->cairocmds, &cmd);
		}
	}

	cmd.layer = 3;
	memcpy(cmd.rgba, pargs->rgba, sizeof(cmd.rgba));
	cmd.text = strdup(txt);
    cmd.x = px;
	cmd.y = py;
	bl_append(ann->cairocmds, &cmd);

    // blank out anything on the lower layers underneath the text.
	/*
	 cairo_save(cairos->shapesmask);
	 cairo_set_source_rgba(cairos->shapesmask, 0, 0, 0, 0);
	 cairo_set_operator(cairos->shapesmask, CAIRO_OPERATOR_SOURCE);
	 cairo_move_to(cairos->shapesmask, l, t);
	 cairo_line_to(cairos->shapesmask, l, b);
	 cairo_line_to(cairos->shapesmask, r, b);
	 cairo_line_to(cairos->shapesmask, r, t);
	 cairo_close_path(cairos->shapesmask);
	 cairo_fill(cairos->shapesmask);
	 cairo_stroke(cairos->shapesmask);
	 cairo_restore(cairos->shapesmask);
	 */
}

static void plot_targets(cairo_t* cairo, plot_args_t* pargs, ann_t* ann) {
	int i;
	double cra, cdec;
	sip_get_radec_center(pargs->wcs, &cra, &cdec);
	
	for (i=0; i<bl_size(ann->targets); i++) {
		target_t* tar = bl_access(ann->targets, i);
		cairocmd_t cmd;
		double px,py;
		double cx,cy;
		double dx,dy, r;
		double ex,ey;
		double ly, ry, tx, bx;
		double distdeg;
		memset(&cmd, 0, sizeof(cmd));

		cmd.layer = 3;
		memcpy(cmd.rgba, pargs->rgba, sizeof(cmd.rgba));

		if (!plotstuff_radec2xy(pargs, tar->ra, tar->dec, &px, &py))
			continue;

		if (px >= 0 && px < pargs->W && py >= 0 && py < pargs->H) {
			// inside the image!
			cmd.type = MARKER;
			cmd.x = px;
			cmd.y = py;
			cmd.marker = pargs->marker;
			cmd.markersize = pargs->markersize;
			bl_append(ann->cairocmds, &cmd);
			cmd.type = TEXT;
			cmd.x = px;
			cmd.y = py;
			cmd.text = strdup(tar->name);
			bl_append(ann->cairocmds, &cmd);
			continue;
		}

		// outside the image: find intersection point.
		cx = pargs->W / 2.0;
		cy = pargs->H / 2.0;

		dx = px - cx;
		dy = py - cy;

		ly = (-(pargs->W/2.0) / dx) * (dy) + cy;
		ry = ( (pargs->W/2.0) / dx) * (dy) + cy;
		bx = (-(pargs->H/2.0) / dy) * (dx) + cx;
		tx = ( (pargs->H/2.0) / dy) * (dx) + cx;
		if (ly >= 0 && ly < pargs->H) {
			ex = 0.0;
			ey = ly;
		} else if (ry >= 0 && ry < pargs->H) {
			ex = pargs->W - 1;
			ey = ry;
		} else if (bx >= 0 && bx < pargs->W) {
			ex = bx;
			ey = 0;
		} else if (tx >= 0 && tx < pargs->W) {
			ex = tx;
			ey = pargs->H - 1;
		} else {
			logverb("None of the edges are in bounds: px,py=(%g,%g); ly=%g, ry=%g, bx=%g, tx=%g\n", px,py,ly,ry,bx,tx);
			continue;
		}
		dx = ex - cx;
		dy = ey - cy;
		r = sqrt(dx*dx + dy*dy);
		
		cmd.type = ARROW;
		cmd.x = (r-100.0) / r * dx + cx;
		cmd.y = (r-100.0) / r * dy + cy;
		cmd.x2 = ex;
		cmd.y2 = ey;
		bl_append(ann->cairocmds, &cmd);

		distdeg = deg_between_radecdeg(cra, cdec, tar->ra, tar->dec);

		cmd.type = TEXT;
		//cmd.text = strdup(tar->name);
		asprintf(&cmd.text, "%s: %.2f deg", tar->name, distdeg);
		bl_append(ann->cairocmds, &cmd);
	}
}

static void plot_ngc(cairo_t* cairo, plot_args_t* pargs, ann_t* ann) {
	double imscale;
	double imsize;
	int i, N;
	double dy = 0;
	cairo_font_extents_t extents;

    double label_offset = 15.0;

	cairo_font_extents(cairo, &extents);
	dy = extents.ascent * 0.5;

	// arcsec/pixel
	imscale = plotstuff_pixel_scale(pargs);
	// arcmin
	imsize = imscale * MIN(pargs->W, pargs->H) / 60.0;

	N = ngc_num_entries();
	logverb("Checking %i NGC/IC objects.\n", N);

	for (i=0; i<N; i++) {
		ngc_entry* ngc = ngc_get_entry_accurate(i);
		char* names;
		double pixrad;
		cairocmd_t cmd;
		double px, py;
		memset(&cmd, 0, sizeof(cmd));

		if (!ngc)
			break;
		if (ngc->size < imsize * ann->ngc_fraction)
			continue;

		if (!plotstuff_radec2xy(pargs, ngc->ra, ngc->dec, &px, &py))
			continue;

		pixrad = 0.5 * ngc->size * 60.0 / imscale;
		// FIXME --  should plot things whose circles will fall within frame.
		if (px < -pixrad || py < -pixrad || px > pargs->W + pixrad || py > pargs->H + pixrad)
			continue;

		names = ngc_get_name_list(ngc, " / ");
		//if (only_messier && !starts_with(sl_get(names, n), "M "))
		printf("%s\n", names);

		cmd.type = CIRCLE;
		cmd.layer = 0;
		cmd.x = px;
		cmd.y = py;
		cmd.radius = pixrad + 1.0;
		memcpy(cmd.rgba, ann->bg_rgba, sizeof(cmd.rgba));
		bl_append(ann->cairocmds, &cmd);

		cmd.radius = pixrad - 1.0;
		bl_append(ann->cairocmds, &cmd);

		cmd.layer = 1;
		cmd.radius = pixrad;
		memcpy(cmd.rgba, pargs->rgba, sizeof(cmd.rgba));
		bl_append(ann->cairocmds, &cmd);

		debug("size: %f arcsec, pix radius: %f pixels\n", ngc->size, pixrad);

		add_text(pargs, ann, cairo, names, px + label_offset, py + dy);
		free(names);

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
	ann_t* ann = calloc(1, sizeof(ann_t));
	ann->cairocmds = bl_new(256, sizeof(cairocmd_t));
	ann->ngc_fraction = 0.02;
	ann->fontname = strdup("DejaVu Sans Mono Book");
	parse_color_rgba("black", ann->bg_rgba);
	ann->fontsize = 14.0;
	ann->targets = bl_new(4, sizeof(target_t));
	return ann;
}

int plot_annotations_plot(const char* cmd, cairo_t* cairo,
							 plot_args_t* pargs, void* baton) {
	ann_t* ann = (ann_t*)baton;
	int i;
	int layer;
	bool morelayers;

	// Set fonts, etc, before calling plotting routines
	cairo_select_font_face(cairo, ann->fontname, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
	cairo_set_font_size(cairo, ann->fontsize);

	plot_ngc(cairo, pargs, ann);

	if (bl_size(ann->targets))
		plot_targets(cairo, pargs, ann);

	morelayers = TRUE;
	for (layer=0;; layer++) {
		if (!morelayers)
			break;
		morelayers = FALSE;
		for (i=0; i<bl_size(ann->cairocmds); i++) {
			cairocmd_t* cmd = bl_access(ann->cairocmds, i);
			if (cmd->layer > layer)
				morelayers = TRUE;
			if (cmd->layer != layer)
				continue;
			cairo_set_rgba(cairo, cmd->rgba);
			switch (cmd->type) {
			case CIRCLE:
				cairo_move_to(cairo, cmd->x + cmd->radius, cmd->y);
				cairo_arc(cairo, cmd->x, cmd->y, cmd->radius, 0, 2*M_PI);
				break;
			case MARKER:
				cairoutils_draw_marker(cairo, cmd->marker, cmd->x, cmd->y, cmd->markersize);
				break;
			case TEXT:
				cairo_move_to(cairo, cmd->x, cmd->y);
				cairo_show_text(cairo, cmd->text);
				break;
			case LINE:
			case ARROW:
				cairo_move_to(cairo, cmd->x, cmd->y);
				cairo_line_to(cairo, cmd->x2, cmd->y2);
				{
					double dx = cmd->x - cmd->x2;
					double dy = cmd->y - cmd->y2;
					double angle = atan2(dy, dx);
					double dang = 30. * M_PI/180.0;
					double arrowlen = 20;
					cairo_line_to(cairo,
								  cmd->x2 + cos(angle+dang)*arrowlen,
								  cmd->y2 + sin(angle+dang)*arrowlen);
					cairo_move_to(cairo, cmd->x2, cmd->y2);
					cairo_line_to(cairo,
								  cmd->x2 + cos(angle-dang)*arrowlen,
								  cmd->y2 + sin(angle-dang)*arrowlen);
				}
				break;
			case RECTANGLE:
				ERROR("Unimplemented!");
				return -1;
			}
			cairo_stroke(cairo);
		}
	}
	for (i=0; i<bl_size(ann->cairocmds); i++) {
		cairocmd_t* cmd = bl_access(ann->cairocmds, i);
		free(cmd->text);
	}
	bl_remove_all(ann->cairocmds);

	return 0;
}

int plot_annotations_command(const char* cmd, const char* cmdargs,
							 plot_args_t* pargs, void* baton) {
	ann_t* ann = (ann_t*)baton;
	if (streq(cmd, "annotations_fontsize")) {
		ann->fontsize = atoi(cmdargs);
	} else if (streq(cmd, "annotations_font")) {
		free(ann->fontname);
		ann->fontname = strdup(cmdargs);
	} else if (streq(cmd, "annotations_bgcolor")) {
		parse_color_rgba(cmdargs, ann->bg_rgba);
	} else if (streq(cmd, "annotations_target")) {
		target_t tar;
		sl* args = sl_split(NULL, cmdargs, " ");
		memset(&tar, 0, sizeof(target_t));
		if (sl_size(args) == 3) {
			tar.ra = atof(sl_get(args, 0));
			tar.dec = atof(sl_get(args, 1));
			tar.name = strdup(sl_get(args, 2));
			bl_append(ann->targets, &tar);
		} else {
			ERROR("Need RA,Dec,name");
			return -1;
		}
	} else if (streq(cmd, "annotations_targetname")) {
		target_t tar;
		char* name = cmdargs;
		ngc_entry* e = ngc_get_entry_named(name);
		if (!e) {
			ERROR("Failed to find target named \"%s\"", name);
			return -1;
		}
		tar.name = ngc_get_name_list(e, "/");
		logmsg("Found %s\n", tar.name);
		tar.ra = e->ra;
		tar.dec = e->dec;
		bl_append(ann->targets, &tar);
	} else {
		ERROR("Unknown command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_annotations_free(plot_args_t* args, void* baton) {
	ann_t* ann = (ann_t*)baton;
	bl_free(ann->cairocmds);
	free(ann->fontname);
	free(ann);
}



