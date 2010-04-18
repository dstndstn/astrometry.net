/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009, 2010 Dustin Lang.

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

// Avoid *nasty* problem when 'bool' gets redefined (by ppm.h) to be 4 bytes!
#include "an-bool.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/param.h>

#include <cairo.h>
#include <cairo-pdf.h>
#ifndef ASTROMETRY_NO_PPM
#include <ppm.h>
#endif

#include "plotstuff.h"
#include "plotfill.h"
#include "plotxy.h"
#include "plotimage.h"
#include "plotannotations.h"
#include "plotgrid.h"
#include "plotoutline.h"
#include "plotindex.h"

#include "sip_qfits.h"
#include "sip-utils.h"
#include "sip.h"
#include "cairoutils.h"
#include "starutil.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "anwcs.h"

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
	int layer;
	double x, y;
	float rgba[4];
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

int plotstuff_get_radec_center_and_radius(plot_args_t* pargs,
										  double* p_ra, double* p_dec, double* p_radius) {
	if (!pargs->wcs)
		return -1;
	return anwcs_get_radec_center_and_radius(pargs->wcs, p_ra, p_dec, p_radius);
}

int plotstuff_append_doubles(const char* str, dl* lst) {
	int i;
	sl* strs = sl_split(NULL, str, " ");
	for (i=0; i<sl_size(strs); i++)
		dl_append(lst, atof(sl_get(strs, i)));
	sl_free2(strs);
	return 0;
}

int plot_line_constant_ra(plot_args_t* pargs, double ra, double dec1, double dec2) {
	double decstep;
	double dec;
	double s;
	assert(pargs->wcs);
	decstep = arcsec2deg(anwcs_pixel_scale(pargs->wcs) * pargs->linestep);
	s = 1.0;
	if (dec1 > dec2)
		s = -1;
	for (dec=dec1; (s*dec)<=(s*dec2); dec+=(decstep*s)) {
		double x, y;
		if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y))
			continue;
		if (dec == dec1)
			cairo_move_to(pargs->cairo, x, y);
		else
			cairo_line_to(pargs->cairo, x, y);
	}
	return 0;
}

int plot_line_constant_dec(plot_args_t* pargs, double dec, double ra1, double ra2) {
	double rastep;
	double ra;
	double f;
	double s;
	assert(pargs->wcs);
	rastep = arcsec2deg(anwcs_pixel_scale(pargs->wcs) * pargs->linestep);
	f = cos(deg2rad(dec));
	rastep /= MAX(0.1, f);
	s = 1.0;
	if (ra1 > ra2)
		s = -1.0;
	for (ra=ra1; (s*ra)<=(s*ra2); ra+=(rastep*s)) {
		double x, y;
		if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y))
			continue;
		if (ra == ra1)
			cairo_move_to(pargs->cairo, x, y);
		else
			cairo_line_to(pargs->cairo, x, y);
	}
	return 0;
}

int parse_color(const char* color, float* r, float* g, float* b, float* a) {
	if (a) *a = 1.0;
	return (cairoutils_parse_rgba(color, r, g, b, a) &&
			cairoutils_parse_color(color, r, g, b));
}

int parse_color_rgba(const char* color, float* rgba) {
	return parse_color(color, rgba, rgba+1, rgba+2, rgba+3);
}

void cairo_set_rgba(cairo_t* cairo, const float* rgba) {
	cairo_set_source_rgba(cairo, rgba[0], rgba[1], rgba[2], rgba[3]);
}

int cairo_set_color(cairo_t* cairo, const char* color) {
	float rgba[4];
	int res;
	res = parse_color_rgba(color, rgba);
	if (res) {
		ERROR("Failed to parse color \"%s\"", color);
		return res;
	}
	cairo_set_rgba(cairo, rgba);
	return res;
}

void plotstuff_builtin_apply(cairo_t* cairo, plot_args_t* args) {
	cairo_set_rgba(cairo, args->rgba);
	cairo_set_line_width(cairo, args->lw);
	cairo_set_operator(cairo, args->op);
	cairo_set_font_size(cairo, args->fontsize);
}

static void* plot_builtin_init(plot_args_t* args) {
	parse_color_rgba("gray", args->rgba);
	parse_color_rgba("black", args->bg_rgba);
	args->text_bg_layer = 2;
	args->text_fg_layer = 3;
	args->marker_fg_layer = 3;
	args->bg_lw = 3.0;
	args->lw = 1.0;
	args->marker = CAIROUTIL_MARKER_CIRCLE;
	args->markersize = 5.0;
	args->linestep = 10;
	args->op = CAIRO_OPERATOR_OVER;
	args->fontsize = 20;
	args->cairocmds = bl_new(256, sizeof(cairocmd_t));
	args->label_offset_x = 15.0;
	args->label_offset_y = 0.0;
	return NULL;
}

static int plot_builtin_init2(plot_args_t* pargs, void* baton) {
	plotstuff_builtin_apply(pargs->cairo, pargs);
	// Inits that aren't in "plot_builtin"
	cairo_set_antialias(pargs->cairo, CAIRO_ANTIALIAS_GRAY);
	return 0;
}

int plotstuff_set_markersize(plot_args_t* pargs, double ms) {
	pargs->markersize = ms;
	return 0;
}

int plotstuff_set_marker(plot_args_t* pargs, const char* name) {
	int m = cairoutils_parse_marker(name);
	if (m == -1) {
		ERROR("Failed to parse plot_marker \"%s\"", name);
		return -1;
	}
	pargs->marker = m;
	return 0;
}

int plotstuff_set_size(plot_args_t* pargs, int W, int H) {
	pargs->W = W;
	pargs->H = H;
	return 0;
}

static int plot_builtin_command(const char* cmd, const char* cmdargs,
								plot_args_t* pargs, void* baton) {
	if (streq(cmd, "plot_color")) {
		if (parse_color_rgba(cmdargs, pargs->rgba)) {
			ERROR("Failed to parse plot_color: \"%s\"", cmdargs);
			return -1;
		}
	} else if (streq(cmd, "plot_bgcolor")) {
		if (parse_color_rgba(cmdargs, pargs->bg_rgba)) {
			ERROR("Failed to parse plot_bgcolor: \"%s\"", cmdargs);
			return -1;
		}
	} else if (streq(cmd, "plot_fontsize")) {
		pargs->fontsize = atof(cmdargs);
	} else if (streq(cmd, "plot_alpha")) {
		// FIXME -- add checking.
		pargs->rgba[3] = atof(cmdargs);
	} else if (streq(cmd, "plot_op")) {
		if (streq(cmdargs, "add")) {
			pargs->op = CAIRO_OPERATOR_ADD;
		} else if (streq(cmdargs, "reset")) {
			pargs->op = CAIRO_OPERATOR_OVER;
		} else {
			ERROR("Didn't understand op: %s", cmdargs);
			return -1;
		}
	} else if (streq(cmd, "plot_lw")) {
		pargs->lw = atof(cmdargs);
	} else if (streq(cmd, "plot_bglw")) {
		pargs->bg_lw = atof(cmdargs);
	} else if (streq(cmd, "plot_marker")) {
		if (plotstuff_set_marker(pargs, cmdargs)) {
			return -1;
		}
	} else if (streq(cmd, "plot_markersize")) {
		pargs->markersize = atof(cmdargs);
	} else if (streq(cmd, "plot_size")) {
		int W, H;
		if (sscanf(cmdargs, "%i %i", &W, &H) != 2) {
			ERROR("Failed to parse plot_size args \"%s\"", cmdargs);
			return -1;
		}
		plotstuff_set_size(pargs, W, H);
	} else if (streq(cmd, "plot_wcs")) {
		pargs->wcs = anwcs_open(cmdargs, 0);
		if (!pargs->wcs) {
			ERROR("Failed to read WCS file \"%s\"", cmdargs);
			return -1;
		}
	} else if (streq(cmd, "plot_wcs_box")) {
		float ra, dec, width;
		tan_t tanwcs;
		double scale;
		if (sscanf(cmdargs, "%f %f %f", &ra, &dec, &width) != 3) {
			ERROR("Failed to parse plot_wcs_box args \"%s\"", cmdargs);
			return -1;
		}
		logverb("Setting WCS to a box centered at (%g,%g) with width %g deg.\n", ra, dec, width);
		if (pargs->wcs)
			anwcs_free(pargs->wcs);

		tanwcs.crval[0] = ra;
		tanwcs.crval[1] = dec;
		tanwcs.crpix[0] = pargs->W / 2.0;
		tanwcs.crpix[1] = pargs->H / 2.0;
		scale = width / (double)pargs->W;
		tanwcs.cd[0][0] = -scale;
		tanwcs.cd[1][0] = 0;
		tanwcs.cd[0][1] = 0;
		tanwcs.cd[1][1] = -scale;
		tanwcs.imagew = pargs->W;
		tanwcs.imageh = pargs->H;
		pargs->wcs = anwcs_new_tan(&tanwcs);

	} else if (streq(cmd, "plot_wcs_setsize")) {
		assert(pargs->wcs);
		plotstuff_set_size(pargs, (int)ceil(anwcs_imagew(pargs->wcs)), (int)ceil(anwcs_imageh(pargs->wcs)));
	} else {
		ERROR("Did not understand command: \"%s\"", cmd);
		return -1;
	}
	if (pargs->cairo)
		plotstuff_builtin_apply(pargs->cairo, pargs);
	return 0;
}

static void add_cmd(plot_args_t* pargs, cairocmd_t* cmd) {
	bl_append(pargs->cairocmds, cmd);
}

void plotstuff_stack_marker(plot_args_t* pargs, double x, double y) {
	cairocmd_t cmd;
	// BG marker?
	memset(&cmd, 0, sizeof(cmd));
	cmd.layer = pargs->marker_fg_layer;
	cmd.type = MARKER;
	cmd.x = x;
	cmd.y = y;
	cmd.marker = pargs->marker;
	cmd.markersize = pargs->markersize;
	add_cmd(pargs, &cmd);
}

void plotstuff_stack_arrow(plot_args_t* pargs, double x, double y,
						   double x2, double y2) {
	cairocmd_t cmd;
	// BG?
	memset(&cmd, 0, sizeof(cmd));
	cmd.layer = pargs->marker_fg_layer;
	cmd.type = ARROW;
	cmd.x = x;
	cmd.y = y;
	cmd.x2 = x2;
	cmd.y2 = y2;
	add_cmd(pargs, &cmd);
}

void plotstuff_stack_text(plot_args_t* pargs, cairo_t* cairo,
						  const char* txt, double px, double py) {
    cairo_text_extents_t textents;
    double l,r,t,b;
    double margin = 2.0;
    int dx, dy;
	cairocmd_t cmd;
	memset(&cmd, 0, sizeof(cmd));

	px += pargs->label_offset_x;
	py += pargs->label_offset_y;

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
	cmd.layer = pargs->text_bg_layer;
	memcpy(cmd.rgba, pargs->bg_rgba, sizeof(cmd.rgba));
    for (dy=-1; dy<=1; dy++) {
        for (dx=-1; dx<=1; dx++) {
			cmd.text = strdup(txt);
			cmd.x = px + dx;
			cmd.y = py + dy;
			add_cmd(pargs, &cmd);
		}
	}

	cmd.layer = pargs->text_fg_layer;
	memcpy(cmd.rgba, pargs->rgba, sizeof(cmd.rgba));
	cmd.text = strdup(txt);
    cmd.x = px;
	cmd.y = py;
	add_cmd(pargs, &cmd);
}

int plotstuff_plot_stack(plot_args_t* pargs, cairo_t* cairo) {
	int i;
	int layer;
	bool morelayers;

	morelayers = TRUE;
	for (layer=0;; layer++) {
		if (!morelayers)
			break;
		morelayers = FALSE;
		for (i=0; i<bl_size(pargs->cairocmds); i++) {
			cairocmd_t* cmd = bl_access(pargs->cairocmds, i);
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
				cairo_move_to(cairo, cmd->x, cmd->y);
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
	for (i=0; i<bl_size(pargs->cairocmds); i++) {
		cairocmd_t* cmd = bl_access(pargs->cairocmds, i);
		free(cmd->text);
	}
	bl_remove_all(pargs->cairocmds);

	return 0;
}

static void plot_builtin_free(plot_args_t* pargs, void* baton) {
	free(pargs->wcs);
	bl_free(pargs->cairocmds);
}

static const plotter_t builtin = { "plot", plot_builtin_init, plot_builtin_init2, plot_builtin_command, NULL, plot_builtin_free, NULL };

int parse_image_format(const char* fmt) {
	if (streq(fmt, "png")) {
		return PLOTSTUFF_FORMAT_PNG;
	} else if (streq(fmt, "jpg") || streq(fmt, "jpeg")) {
		return PLOTSTUFF_FORMAT_JPG;
	} else if (streq(fmt, "ppm")) {
		return PLOTSTUFF_FORMAT_PPM;
	} else if (streq(fmt, "pdf")) {
		return PLOTSTUFF_FORMAT_PDF;
	} else if (streq(fmt, "fits")) {
		return PLOTSTUFF_FORMAT_FITS;
	}
	ERROR("Unknown image format \"%s\"", fmt);
	return -1;
}

int plotstuff_set_color(plot_args_t* pargs, const char* name) {
	return parse_color_rgba(name, pargs->rgba);
}

int plotstuff_set_rgba(plot_args_t* pargs, const float* rgba) {
	pargs->rgba[0] = rgba[0];
	pargs->rgba[1] = rgba[1];
	pargs->rgba[2] = rgba[2];
	pargs->rgba[3] = rgba[3];
	return 0;
}

/* All render layers must go in here */
static plotter_t plotters[8];

int plotstuff_init(plot_args_t* pargs) {
	int i, NR;

	// ?
	memset(pargs, 0, sizeof(plot_args_t));

	plotters[0] = builtin;
	plotters[1] = plotter_fill;
	plotters[2] = plotter_xy;
	plotters[3] = plotter_image;
	plotters[4] = plotter_annotations;
	plotters[5] = plotter_grid;
	plotters[6] = plotter_outline;
	plotters[7] = plotter_index;

	NR = sizeof(plotters) / sizeof(plotter_t);
	// First init
	for (i=0; i<NR; i++)
		plotters[i].baton = plotters[i].init(pargs);
	return 0;
}

int plotstuff_init2(plot_args_t* pargs) {
	int i, NR;

	logverb("Creating drawing surface (%ix%i)\n", pargs->W, pargs->H);
	// Allocate cairo surface
	switch (pargs->outformat) {
	case PLOTSTUFF_FORMAT_PDF:
		if (pargs->outfn) {
			pargs->fout = fopen(pargs->outfn, "wb");
			if (!pargs->fout) {
				SYSERROR("Failed to open output file \"%s\"", pargs->outfn);
				return -1;
			}
		}
		pargs->target = cairo_pdf_surface_create_for_stream(cairoutils_file_write_func, pargs->fout, pargs->W, pargs->H);
		break;
	case PLOTSTUFF_FORMAT_JPG:
	case PLOTSTUFF_FORMAT_PPM:
	case PLOTSTUFF_FORMAT_PNG:
		pargs->target = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, pargs->W, pargs->H);
		break;
	default:
		ERROR("Unknown output format %i", pargs->outformat);
		return -1;
		break;
	}
	pargs->cairo = cairo_create(pargs->target);

	NR = sizeof(plotters) / sizeof(plotter_t);
	for (i=0; i<NR; i++) {
		if (plotters[i].init2 &&
			plotters[i].init2(pargs, plotters[i].baton)) {
			ERROR("Plot initializer failed");
			exit(-1);
		}
	}
	return 0;
}

void* plotstuff_get_config(plot_args_t* pargs, const char* name) {
	int i, NR;
	NR = sizeof(plotters) / sizeof(plotter_t);
	for (i=0; i<NR; i++) {
		if (streq(plotters[i].name, name))
			return plotters[i].baton;
	}
	return NULL;
}

double plotstuff_pixel_scale(plot_args_t* pargs) {
	if (!pargs->wcs) {
		ERROR("plotstuff_pixel_scale: No WCS defined!");
		return 0.0;
	}
	return anwcs_pixel_scale(pargs->wcs);
}

bool plotstuff_radec2xy(plot_args_t* pargs, double ra, double dec,
						double* x, double* y) {
	if (!pargs->wcs) {
		ERROR("No WCS defined!");
		return FALSE;
	}
	return (anwcs_radec2pixelxy(pargs->wcs, ra, dec, x, y) ? FALSE : TRUE);
}

bool plotstuff_radec_is_inside_image(plot_args_t* pargs, double ra, double dec) {
	if (!pargs->wcs) {
		ERROR("No WCS defined!");
		return FALSE;
	}
	return anwcs_radec_is_inside_image(pargs->wcs, ra, dec);
}

void plotstuff_get_radec_bounds(const plot_args_t* pargs, int stepsize,
								double* pramin, double* pramax,
								double* pdecmin, double* pdecmax) {
	if (!pargs->wcs) {
		ERROR("No WCS defined!");
		return;
	}
	return anwcs_get_radec_bounds(pargs->wcs, stepsize, pramin, pramax, pdecmin, pdecmax);
}

int
ATTRIB_FORMAT(printf,2,3)
plotstuff_run_commandf(plot_args_t* pargs, const char* format, ...) {
	char* str;
    va_list va;
	int rtn;
    va_start(va, format);
    if (vasprintf(&str, format, va) == -1) {
		ERROR("Failed to allocate temporary string to hold command");
		return -1;
	}
	rtn = plotstuff_run_command(pargs, str);
    va_end(va);
	return rtn;
}

int plotstuff_run_command(plot_args_t* pargs, const char* cmd) {
	int i, NR;
	bool matched = FALSE;
	if (!cmd || (strlen(cmd) == 0) || (cmd[0] == '#')) {
		return 0;
	}
	NR = sizeof(plotters) / sizeof(plotter_t);
	for (i=0; i<NR; i++) {
		if (streq(cmd, plotters[i].name)) {
			if (!pargs->cairo) {
				if (plotstuff_init2(pargs)) {
					return -1;
				}
			}
			if (plotters[i].doplot) {
				if (plotters[i].doplot(cmd, pargs->cairo, pargs, plotters[i].baton)) {
					ERROR("Plotter \"%s\" failed on command \"%s\"", plotters[i].name, cmd);
					return -1;
				}
			}
		} else if (starts_with(cmd, plotters[i].name)) {
			char* cmdcmd;
			char* cmdargs;
			if (!split_string_once(cmd, " ", &cmdcmd, &cmdargs)) {
				//ERROR("Failed to split command \"%s\" into words\n", cmd);
				//return -1;
				cmdcmd = strdup(cmd);
				cmdargs = NULL;
			}
			logmsg("Command \"%s\", args \"%s\"\n", cmdcmd, cmdargs);
			if (plotters[i].command(cmdcmd, cmdargs, pargs, plotters[i].baton)) {
				ERROR("Plotter \"%s\" failed on command \"%s\"", plotters[i].name, cmd);
				return -1;
			}
			free(cmdcmd);
			free(cmdargs);
		} else
			continue;
		matched = TRUE;
		break;
	}
	if (!matched) {
		ERROR("Did not find a plotter for command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

int plotstuff_read_and_run_command(plot_args_t* pargs, FILE* f) {
	char* cmd;
	int rtn;
	cmd = read_string_terminated(stdin, "\n\r\0", 3, FALSE);
	logverb("command: \"%s\"\n", cmd);
	if (!cmd || feof(f)) {
		free(cmd);
		return -1;
	}
	rtn = plotstuff_run_command(pargs, cmd);
	free(cmd);
	return rtn;
}

int plotstuff_output(plot_args_t* pargs) {
	switch (pargs->outformat) {
	case PLOTSTUFF_FORMAT_PDF:
		cairo_surface_flush(pargs->target);
		cairo_surface_finish(pargs->target);
		cairoutils_surface_status_errors(pargs->target);
		cairoutils_cairo_status_errors(pargs->cairo);
		if (pargs->outfn) {
			if (fclose(pargs->fout)) {
				SYSERROR("Failed to close output file \"%s\"", pargs->outfn);
				return -1;
			}
		}
		break;

	case PLOTSTUFF_FORMAT_JPG:
	case PLOTSTUFF_FORMAT_PPM:
	case PLOTSTUFF_FORMAT_PNG:
	case PLOTSTUFF_FORMAT_MEMIMG:
		{
			int res;
			unsigned char* img = cairo_image_surface_get_data(pargs->target);
			// Convert image for output...
			cairoutils_argb32_to_rgba(img, pargs->W, pargs->H);
			if (pargs->outformat == PLOTSTUFF_FORMAT_MEMIMG) {
				pargs->outimage = img;
				res = 0;
				img = NULL;
			} else if (pargs->outformat == PLOTSTUFF_FORMAT_JPG) {
				res = cairoutils_write_jpeg(pargs->outfn, img, pargs->W, pargs->H);
			} else if (pargs->outformat == PLOTSTUFF_FORMAT_PPM) {
				res = cairoutils_write_ppm(pargs->outfn, img, pargs->W, pargs->H);
			} else if (pargs->outformat == PLOTSTUFF_FORMAT_PNG) {
				res = cairoutils_write_png(pargs->outfn, img, pargs->W, pargs->H);
			} else {
				res=-1; // for gcc
				assert(0);
			}
			if (res)
				ERROR("Failed to write output image");
			return res;
		}
		break;
	default:
		ERROR("Unknown output format.");
		return -1;
	}
	return 0;
}

void plotstuff_free(plot_args_t* pargs) {
	int i, NR;
	NR = sizeof(plotters) / sizeof(plotter_t);
	for (i=0; i<NR; i++) {
		plotters[i].free(pargs, plotters[i].baton);
	}
	cairo_destroy(pargs->cairo);
	cairo_surface_destroy(pargs->target);
}

