/*
  This file is part of the Astrometry.net suite.
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
#include <string.h>
#include <math.h>
#include <sys/param.h>

#include "plotimage.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "sip_qfits.h"
#include "log.h"
#include "errors.h"
#include "anwcs.h"

const plotter_t plotter_image = {
	.name = "image",
	.init = plot_image_init,
	.command = plot_image_command,
	.doplot = plot_image_plot,
	.free = plot_image_free
};

plotimage_t* plot_image_get(plot_args_t* pargs) {
	return plotstuff_get_config(pargs, "image");
}

void* plot_image_init(plot_args_t* plotargs) {
	plotimage_t* args = calloc(1, sizeof(plotimage_t));
	args->gridsize = 50;
	args->alpha = 1;
	args->image_null = 1.0 / 0.0;
	//args->scalex = args->scaley = 1.0;
	return args;
}

//void plot_image_rgba_data(cairo_t* cairo, unsigned char* img, int W, int H, double alpha) 

void plot_image_rgba_data(cairo_t* cairo, plotimage_t* args) {
	cairo_surface_t* thissurf;
	cairo_pattern_t* pat;
	cairoutils_rgba_to_argb32(args->img, args->W, args->H);
	thissurf = cairo_image_surface_create_for_data(args->img, CAIRO_FORMAT_ARGB32, args->W, args->H, args->W*4);
	pat = cairo_pattern_create_for_surface(thissurf);
	cairo_save(cairo);
	cairo_set_source(cairo, pat);
	//cairo_scale(cairo, args->scalex, args->scaley);
	cairo_paint_with_alpha(cairo, args->alpha);
	cairo_pattern_destroy(pat);
	cairo_surface_destroy(thissurf);
	cairo_restore(cairo);
}

void plot_image_wcs(cairo_t* cairo, unsigned char* img, int W, int H,
					plot_args_t* pargs, plotimage_t* args) {
	cairo_surface_t* thissurf;
	cairo_pattern_t* pat;
	cairo_matrix_t mat;
	int i,j;
	double *xs, *ys;
	int NX, NY;
	double x,y;

	cairoutils_rgba_to_argb32(img, W, H);
	thissurf = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, W, H, W*4);

	cairoutils_surface_status_errors(thissurf);
	cairoutils_cairo_status_errors(cairo);

	if (args->alpha != 1.0) {
		unsigned char a = MIN(255, MAX(0, args->alpha * 255));
		for (i=0; i<(W*H); i++)
			img[i*4+3] = a;
	}
	pat = cairo_pattern_create_for_surface(thissurf);
	cairoutils_cairo_status_errors(cairo);

	assert(args->gridsize >= 1);
	NX = 1 + ceil(W / args->gridsize);
	NY = 1 + ceil(H / args->gridsize);
	xs = malloc(NX*NY * sizeof(double));
	ys = malloc(NX*NY * sizeof(double));

	// FIXME -- NEAREST is good when we're zooming in on individual pixels;
	// some smoothing is necessary if we're zoomed out.  Should probably
	// resample image in this case, since I doubt cairo is very smart in this case.
	cairo_pattern_set_filter(pat, CAIRO_FILTER_GOOD);
							 //CAIRO_FILTER_NEAREST);
	for (j=0; j<NY; j++) {
		double ra,dec;
		y = MIN(j * args->gridsize, H-1);
		for (i=0; i<NX; i++) {
			double ox,oy;
			bool ok;
			x = MIN(i * args->gridsize, W-1);
			anwcs_pixelxy2radec(args->wcs, x+1, y+1, &ra, &dec);
			ok = plotstuff_radec2xy(pargs, ra, dec, &ox, &oy);
			xs[j*NX+i] = ox-1;
			ys[j*NX+i] = oy-1;
			debug("image (%g,%g) -> plot (%g,%g)\n", x, y, xs[j*NX+i], ys[j*NX+i]);
		}
	}
	cairo_save(cairo);
	cairo_set_source(cairo, pat);
    //cairo_set_source_rgb(cairo, 1,0,0);
	for (j=0; j<(NY-1); j++) {
		for (i=0; i<(NX-1); i++) {
			int aa = j*NX + i;
			int ab = aa + 1;
			int ba = aa + NX;
			int bb = aa + NX + 1;
            double midx = (xs[aa] + xs[ab] + xs[bb] + xs[ba])*0.25;
            double midy = (ys[aa] + ys[ab] + ys[bb] + ys[ba])*0.25;

            double xlo,xhi,ylo,yhi;
            ylo = MIN(j     * args->gridsize, H-1);
            yhi = MIN((j+1) * args->gridsize, H-1);
			xlo = MIN(i     * args->gridsize, W-1);
			xhi = MIN((i+1) * args->gridsize, W-1);

            cairo_move_to(cairo,
                          0.5 + xs[aa]+0.5*(xs[aa] >= midx ? 1 : -1),
                          0.5 + ys[aa]+0.5*(ys[aa] >= midy ? 1 : -1));
            cairo_line_to(cairo,
                          0.5 + xs[ab]+0.5*(xs[ab] >= midx ? 1 : -1),
                          0.5 + ys[ab]+0.5*(ys[ab] >= midy ? 1 : -1));
            cairo_line_to(cairo,
                          0.5 + xs[bb]+0.5*(xs[bb] >= midx ? 1 : -1),
                          0.5 + ys[bb]+0.5*(ys[bb] >= midy ? 1 : -1));
            cairo_line_to(cairo,
                          0.5 + xs[ba]+0.5*(xs[ba] >= midx ? 1 : -1),
                          0.5 + ys[ba]+0.5*(ys[ba] >= midy ? 1 : -1));
			cairo_close_path(cairo);
			cairo_matrix_init(&mat,
							  (xs[ab]-xs[aa])/(xhi-xlo),
							  (ys[ab]-ys[aa])/(yhi-ylo),
							  (xs[ba]-xs[aa])/(xhi-xlo),
							  (ys[ba]-ys[aa])/(yhi-ylo),
							  xs[0], ys[0]);
			cairo_matrix_invert(&mat);
			cairo_pattern_set_matrix(pat, &mat);

			cairo_fill(cairo);
		}
	}
	/* Grid:
	 cairo_set_source_rgb(cairo, 1,0,0);
	 for (j=0; j<(NY-1); j++) {
	 for (i=0; i<(NX-1); i++) {
	 int aa = j*NX + i;
	 int ab = aa + 1;
	 int ba = aa + NX;
	 int bb = aa + NX + 1;
	 cairo_move_to(cairo, xs[aa], ys[aa]);
	 cairo_line_to(cairo, xs[ab], ys[ab]);
	 cairo_line_to(cairo, xs[bb], ys[bb]);
	 cairo_line_to(cairo, xs[ba], ys[ba]);
	 cairo_close_path(cairo);
	 cairo_stroke(cairo);
	 }
	 }
	 {
	 int aa = 0;
	 int ab = 1;
	 int ba = NX;
	 int bb = NX + 1;
	 cairo_set_source_rgb(cairo, 0,1,0);
	 cairo_move_to(cairo, xs[aa], ys[aa]);
	 cairo_line_to(cairo, xs[ab], ys[ab]);
	 cairo_line_to(cairo, xs[bb], ys[bb]);
	 cairo_line_to(cairo, xs[ba], ys[ba]);
	 cairo_close_path(cairo);
	 cairo_move_to(cairo, 0, 0);
	 cairo_line_to(cairo, 0, args->gridsize);
	 cairo_line_to(cairo, args->gridsize, args->gridsize);
	 cairo_line_to(cairo, args->gridsize, 0);
	 cairo_close_path(cairo);
	 cairo_stroke(cairo);
	 }
	 */

	free(xs);
	free(ys);

	cairo_pattern_destroy(pat);
	cairo_surface_destroy(thissurf);
	cairo_restore(cairo);
}

static unsigned char* read_fits_image(plotimage_t* args) {
	float* fimg;
	qfitsloader ld;
	unsigned char* img;
	int i,j;
	float offset, scale;

	ld.filename = args->fn;
	ld.xtnum = args->fitsext;
	ld.pnum = args->fitsplane;
	ld.map = 1;
	ld.ptype = PTYPE_FLOAT;
		
	if (qfitsloader_init(&ld)) {
		ERROR("qfitsloader_init() failed");
		return NULL;
	}
	if (qfits_loadpix(&ld)) {
		ERROR("qfits_loadpix() failed");
		return NULL;
	}
	args->W = ld.lx;
	args->H = ld.ly;
	fimg = ld.fbuf;

	if (args->image_low == 0 && args->image_high == 0) {
		offset = 0.0;
		scale = 1.0;
	} else {
		offset = args->image_low;
		scale = 255.0 / (args->image_high - args->image_low);
		logmsg("Image range %g, %g --> offset %g, scale %g\n", args->image_low, args->image_high, offset, scale);
	}

	img = malloc(args->W * args->H * 4);
	for (j=0; j<args->H; j++) {
		for (i=0; i<args->W; i++) {
			int k;
			unsigned char v;
			double pval = fimg[j*args->W + i];
			k = 4*(j*args->W + i);
			if ((isnan(args->image_null) && isnan(pval)) ||
				(args->image_null == pval)) {
				img[k+0] = 0;
				img[k+1] = 0;
				img[k+2] = 0;
				img[k+3] = 0;
			} else {
				v = MIN(255, MAX(0, (pval - offset) * scale));
				img[k+0] = v;
				img[k+1] = v;
				img[k+2] = v;
				img[k+3] = 255;
			}
		}
	}
	qfitsloader_free_buffers(&ld);
	return img;
}

int plot_image_read(plotimage_t* args) {
	// FIXME -- guess format from filename?
	if (args->format == 0) {
		args->format = guess_image_format_from_filename(args->fn);
		logverb("Guessing format of image from filename: \"%s\" -> %s", args->fn, image_format_name_from_code(args->format));
	}
	switch (args->format) {
	case PLOTSTUFF_FORMAT_JPG:
		args->img = cairoutils_read_jpeg(args->fn, &(args->W), &(args->H));
		break;
	case PLOTSTUFF_FORMAT_PNG:
		args->img = cairoutils_read_png(args->fn, &(args->W), &(args->H));
		break;
	case PLOTSTUFF_FORMAT_PPM:
		args->img = cairoutils_read_ppm(args->fn, &(args->W), &(args->H));
		break;
	case PLOTSTUFF_FORMAT_FITS:
		args->img = read_fits_image(args);
		break;
	case PLOTSTUFF_FORMAT_PDF:
		ERROR("PDF format not supported");
		return -1;
	default:
		ERROR("You must set the image format with \"image_format <png|jpg|ppm>\"");
		return -1;
	}
	return 0;
}

int plot_image_set_filename(plotimage_t* args, const char* fn) {
	free(args->fn);
	args->fn = strdup_safe(fn);
	return 0;
}

int plot_image_plot(const char* command,
					cairo_t* cairo, plot_args_t* pargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	// Plot it!
	if (!args->img) {
		if (plot_image_read(args)) {
			return -1;
		}
	}

	if (pargs->wcs && args->wcs) {
		plot_image_wcs(cairo, args->img, args->W, args->H, pargs, args);
	} else {
		plot_image_rgba_data(cairo, args);
	}
	// ?
	free(args->img);
	args->img = NULL;
	return 0;
}

int plot_image_setsize(plot_args_t* pargs, plotimage_t* args) {
	if (!args->img) {
		if (plot_image_read(args)) {
			return -1;
		}
	}
	plotstuff_set_size(pargs, args->W, args->H);
	//plotstuff_set_size(pargs, args->W * args->scalex, args->H * args->scaley);
	return 0;
}

int plot_image_set_wcs(plotimage_t* args, const char* filename, int ext) {
	if (args->wcs)
		anwcs_free(args->wcs);
	if (streq(filename, "none")) {
		args->wcs = NULL;
	} else {
		args->wcs = anwcs_open(filename, ext);
		if (!args->wcs) {
			ERROR("Failed to read WCS file \"%s\"", filename);
			return -1;
		}
		if (log_get_level() >= LOG_VERB) {
			logverb("Set image WCS to:");
			anwcs_print(args->wcs, stdout);
		}
	}
	return 0;
}

int plot_image_command(const char* cmd, const char* cmdargs,
					   plot_args_t* pargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	if (streq(cmd, "image_file")) {
		plot_image_set_filename(args, cmdargs);
	} else if (streq(cmd, "image_alpha")) {
		args->alpha = atof(cmdargs);
	} else if (streq(cmd, "image_format")) {
		args->format = parse_image_format(cmdargs);
		if (args->format == -1)
			return -1;
	} else if (streq(cmd, "image_setsize")) {
		if (plot_image_setsize(pargs, args))
			return -1;
	} else if (streq(cmd, "image_wcslib")) {
		// force reading WCS using WCSLIB.
		if (args->wcs)
			anwcs_free(args->wcs);
		args->wcs = anwcs_open_wcslib(cmdargs, 0);
		if (!args->wcs) {
			ERROR("Failed to read WCS file \"%s\"", cmdargs);
			return -1;
		}
		if (log_get_level() >= LOG_VERB) {
			logverb("Set image WCS to:");
			anwcs_print(args->wcs, stdout);
		}
	} else if (streq(cmd, "image_wcs")) {
		return plot_image_set_wcs(args, cmdargs, args->fitsext);
	} else if (streq(cmd, "image_ext")) {
		args->fitsext = atoi(cmdargs);
	} else if (streq(cmd, "image_grid")) {
		args->gridsize = atof(cmdargs);
	} else if (streq(cmd, "image_low")) {
		args->image_low = atof(cmdargs);
		logmsg("set image_low %g\n", args->image_low);
	} else if (streq(cmd, "image_null")) {
		args->image_null = atof(cmdargs);
	} else if (streq(cmd, "image_high")) {
		args->image_high = atof(cmdargs);
		logmsg("set image_high %g\n", args->image_high);
	} else {
		ERROR("Did not understand command \"%s\"", cmd);
		return -1;
	}
	return 0;
}

void plot_image_free(plot_args_t* plotargs, void* baton) {
	plotimage_t* args = (plotimage_t*)baton;
	if (args->wcs)
		anwcs_free(args->wcs);
	free(args->fn);
	free(args);
}

