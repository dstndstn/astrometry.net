/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
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

#include "sip_qfits.h"
#include "sip.h"
#include "cairoutils.h"
#include "starutil.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

int plot_line_constant_ra(plot_args_t* pargs, double ra, double dec1, double dec2) {
	double decstep;
	double dec;
	double s;
	assert(pargs->wcs);
	decstep = arcsec2deg(sip_pixel_scale(pargs->wcs) * pargs->linestep);
	s = 1.0;
	if (dec1 > dec2)
		s = -1;
	for (dec=dec1; (s*dec)<=(s*dec2); dec+=(decstep*s)) {
		double x, y;
		bool ok;
		ok = sip_radec2pixelxy(pargs->wcs, ra, dec, &x, &y);
		if (!ok)
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
	rastep = arcsec2deg(sip_pixel_scale(pargs->wcs) * pargs->linestep);
	f = cos(deg2rad(dec));
	rastep /= MAX(0.1, f);
	s = 1.0;
	if (ra1 > ra2)
		s = -1.0;
	for (ra=ra1; (s*ra)<=(s*ra2); ra+=(rastep*s)) {
		double x, y;
		bool ok;
		ok = sip_radec2pixelxy(pargs->wcs, ra, dec, &x, &y);
		if (!ok)
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

static void plot_builtin_apply(cairo_t* cairo, plot_args_t* args) {
	cairo_set_rgba(cairo, args->rgba);
	cairo_set_line_width(cairo, args->lw);
	cairo_set_operator(cairo, args->op);
	cairo_set_font_size(cairo, args->fontsize);
}

static void* plot_builtin_init(plot_args_t* args) {
	args->rgba[0] = 0;
	args->rgba[1] = 0;
	args->rgba[2] = 0;
	args->rgba[3] = 1;
	args->lw = 1.0;
	args->marker = CAIROUTIL_MARKER_CIRCLE;
	args->markersize = 5.0;
	args->linestep = 10;
	args->op = CAIRO_OPERATOR_OVER;
	args->fontsize = 20;
	return NULL;
}

static int plot_builtin_init2(plot_args_t* pargs, void* baton) {
	plot_builtin_apply(pargs->cairo, pargs);
	// Inits that aren't in "plot_builtin"
	cairo_set_antialias(pargs->cairo, CAIRO_ANTIALIAS_GRAY);
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
	} else if (streq(cmd, "plot_marker")) {
		if (plotstuff_set_marker(pargs, cmdargs)) {
			return -1;
		}
	} else if (streq(cmd, "plot_markersize")) {
		pargs->markersize = atof(cmdargs);
	} else if (streq(cmd, "plot_wcs")) {
		pargs->wcs = sip_read_tan_or_sip_header_file_ext(cmdargs, 0, NULL, FALSE);
		if (!pargs->wcs) {
			ERROR("Failed to read WCS file \"%s\"", cmdargs);
			return -1;
		}
	} else if (streq(cmd, "plot_wcs_setsize")) {
		assert(pargs->wcs);
		plotstuff_set_size(pargs, (int)ceil(sip_imagew(pargs->wcs)), (int)ceil(sip_imageh(pargs->wcs)));
	} else {
		ERROR("Did not understand command: \"%s\"", cmd);
		return -1;
	}
	if (pargs->cairo)
		plot_builtin_apply(pargs->cairo, pargs);
	return 0;
}

static void plot_builtin_free(plot_args_t* pargs, void* baton) {
	free(pargs->wcs);
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
	}
	ERROR("Unknown image format \"%s\"", fmt);
	return -1;
}

int plotstuff_set_color(plot_args_t* pargs, const char* name) {
	return parse_color_rgba(name, pargs->rgba);
}

/* All render layers must go in here */
static plotter_t plotters[7];

int plotstuff_init(plot_args_t* pargs) {
	int i, NR;

	plotters[0] = builtin;
	plotters[1] = plotter_fill;
	plotters[2] = plotter_xy;
	plotters[3] = plotter_image;
	plotters[4] = plotter_annotations;
	plotters[5] = plotter_grid;
	plotters[6] = plotter_outline;

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
	for (i=0; i<NR; i++)
		if (streq(plotters[i].name, name))
			return plotters[i].baton;
	return NULL;
}

double plotstuff_pixel_scale(plot_args_t* pargs) {
	if (!pargs->wcs) {
		ERROR("plotstuff_pixel_scale: No WCS defined!");
		return 0.0;
	}
	return sip_pixel_scale(pargs->wcs);
}

int plotstuff_radec2xy(plot_args_t* pargs, double ra, double dec,
					   double* x, double* y) {
	if (!pargs->wcs) {
		ERROR("No WCS defined!");
		return -1;
	}
	return sip_radec2pixelxy(pargs->wcs, ra, dec, x, y);
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
	logmsg("command: \"%s\"\n", cmd);
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
		{
			int res;
			unsigned char* img = cairo_image_surface_get_data(pargs->target);
			// Convert image for output...
			cairoutils_argb32_to_rgba(img, pargs->W, pargs->H);
			if (pargs->outformat == PLOTSTUFF_FORMAT_JPG) {
				res = cairoutils_write_jpeg(pargs->outfn, img, pargs->W, pargs->H);
			} else if (pargs->outformat == PLOTSTUFF_FORMAT_PPM) {
				res = cairoutils_write_ppm(pargs->outfn, img, pargs->W, pargs->H);
			} else if (pargs->outformat == PLOTSTUFF_FORMAT_PNG) {
				res = cairoutils_write_png(pargs->outfn, img, pargs->W, pargs->H);
			} else {
				res=-1; // for gcc
				assert(0);
			}
			//free(img);
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

