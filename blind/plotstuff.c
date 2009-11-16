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


/**

 Reads a plot control file that contains instructions on what to plot
 and how to plot it.

 There are a few plotting module that understand different commands
 and plot different kinds of things.

 Common plotting:

 plot_color <r> <g> <b> <a>
 plot_lw <linewidth>
 plot_marker <marker-shape>
 plot_markersize <radius>

 Image:

 image_file <fn>
 image_format <format>
 image

 Xy:

 xy_file <xylist>
 xy_ext <fits-extension>
 xy_xcol <column-name>
 xy_ycol <column-name>
 xy_xoff <pixel-offset>
 xy_yoff <pixel-offset>
 xy_firstobj <obj-num>
 xy_nobjs <n>
 xy_scale <factor>
 xy_bgcolor <r> <g> <b> <a>
 xy

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

#include "plotxy.h"
#include "plotimage.h"

#include "boilerplate.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hW:H:o:JP";

int parse_color(const char* color, float* r, float* g, float* b, float* a) {
	if (a) *a = 1.0;
	return (cairoutils_parse_rgba(color, r, g, b, a) &&
			cairoutils_parse_color(color, r, g, b));
}


static void plot_builtin_apply(cairo_t* cairo, plot_args_t* args) {
	cairo_set_source_rgba(cairo, args->r, args->g, args->b, args->a);
	cairo_set_line_width(cairo, args->lw);
}

static void* plot_builtin_init(plot_args_t* args) {
	args->r = 0.0;
	args->g = 0.0;
	args->b = 0.0;
	args->a = 1.0;
	args->lw = 1.0;
	args->marker = CAIROUTIL_MARKER_CIRCLE;
	args->markersize = 5.0;
	//plot_builtin_apply(args);
	return NULL;
}

static int plot_builtin_command(const char* command, cairo_t* cairo,
								plot_args_t* plotargs, void* baton) {
	char* cmd;
	char* cmdargs;
	if (!split_string_once(command, " ", &cmd, &cmdargs)) {
		ERROR("Failed to split command \"%s\" into words\n", command);
		return -1;
	}

	logmsg("Command \"%s\", args \"%s\"\n", cmd, cmdargs);

	if (streq(cmd, "plot_color")) {
		if (parse_color(cmdargs, &(plotargs->r), &(plotargs->g), &(plotargs->b), &(plotargs->a))) {
			ERROR("Failed to parse plot_color: \"%s\"", cmdargs);
			return -1;
		}

	} else if (streq(cmd, "plot_alpha")) {
		// FIXME -- add checking.
		plotargs->a = atof(cmdargs);
	} else if (streq(cmd, "plot_lw")) {
		plotargs->lw = atof(cmdargs);
	} else if (streq(cmd, "plot_marker")) {
		int m = cairoutils_parse_marker(cmdargs);
		if (m == -1) {
			ERROR("Failed to parse plot_marker \"%s\"", cmdargs);
			return -1;
		}
		plotargs->marker = m;
	} else if (streq(cmd, "plot_markersize")) {
		plotargs->markersize = atof(cmdargs);
	} else {
		ERROR("Did not understand command: \"%s\"", cmd);
		return -1;
	}
	plot_builtin_apply(cairo, plotargs);
	free(cmd);
	free(cmdargs);
	return 0;
}

static void plot_builtin_free(plot_args_t* args, void* baton) {
}



struct plotter {
	// don't change the order of these fields!
	char* name;
	plot_func_init_t init;
	plot_func_command_t command;
	plot_func_free_t free;
	void* baton;
};
typedef struct plotter plotter_t;

/* All render layers must go in here */
static plotter_t plotters[] = {
	{ "plot", plot_builtin_init, plot_builtin_command, plot_builtin_free, (void*)NULL },
	{ "xy", plot_xy_init, plot_xy_command, plot_xy_free, NULL },
	{ "image", plot_image_init, plot_image_command, plot_image_free, NULL },
};


static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] > output.png\n"
           "  [-o <output-file>] (default: stdout)\n"
           "  [-P]              Write PPM output instead of PNG.\n"
		   "  [-J]              Write PDF output.\n"
		   "  [-W <width>   ]   Width of output image (default: data-dependent).\n"
		   "  [-H <height>  ]   Height of output image (default: data-dependent).\n",
           progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *args[]) {
	int loglvl = LOG_MSG;
	int argchar;
	char* progname = args[0];
    char* outfn = "-";
	int W = 0, H = 0;
    bool ppmoutput = FALSE;
    bool pdfoutput = FALSE;

	cairo_t* cairo;
	cairo_surface_t* target;

	int i;
	int NR;
	plot_args_t plotargs;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 'o':
            outfn = optarg;
            break;
        case 'P':
            ppmoutput = TRUE;
            break;
		case 'J':
			pdfoutput = TRUE;
			break;
		case 'W':
			W = atoi(optarg);
			break;
		case 'H':
			H = atoi(optarg);
			break;
		case 'v':
			loglvl++;
			break;
		case 'h':
			printHelp(progname);
            exit(0);
		case '?':
		default:
			printHelp(progname);
            exit(-1);
		}

	if (optind != argc) {
		printHelp(progname);
		exit(-1);
	}

	assert(W && H);

	log_init(loglvl);

    // log errors to stderr, not stdout.
    errors_log_to(stderr);

	// Allocate cairo surface
	if (pdfoutput) {
		target = cairo_pdf_surface_create_for_stream(cairoutils_file_write_func, stdout, W, H);
	} else {
		target = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
	}
	cairo = cairo_create(target);

	memset(&plotargs, 0, sizeof(plotargs));

	NR = sizeof(plotters) / sizeof(plotter_t);
	for (i=0; i<NR; i++) {
		plotters[i].baton = plotters[i].init(&plotargs);
	}

	plot_builtin_apply(cairo, &plotargs);
	// Inits that aren't in "plot_builtin"
	cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);

	for (;;) {
		char* cmd = read_string_terminated(stdin, "\n\r\0", 3, FALSE);
		int res;
		bool matched = FALSE;
		if (!cmd)
			break;
		for (i=0; i<NR; i++) {
			if (starts_with(cmd, plotters[i].name)) {
				res = plotters[i].command(cmd, cairo, &plotargs, plotters[i].baton);
				if (res) {
					ERROR("Plotter \"%s\" failed on command \"%s\"", plotters[i].name, cmd);
					exit(-1);
				}
				matched = TRUE;
			}
			if (matched)
				break;
		}
		if (!matched) {
			ERROR("Did not find a plotter for command \"%s\"", cmd);
			exit(-1);
		}
		free(cmd);
	}

	for (i=0; i<NR; i++) {
		plotters[i].free(&plotargs, plotters[i].baton);
	}


	if (pdfoutput) {
		cairo_surface_flush(target);
		cairo_surface_finish(target);
		cairoutils_surface_status_errors(target);
		cairoutils_cairo_status_errors(cairo);
	} else {
		unsigned char* img = cairo_image_surface_get_data(target);
		// Convert image for output...
		cairoutils_argb32_to_rgba(img, W, H);
		if (ppmoutput) {
			if (cairoutils_write_ppm(outfn, img, W, H)) {
				fprintf(stderr, "Failed to write PPM.\n");
				exit(-1);
			}
		} else {
			// PNG
			if (cairoutils_write_png(outfn, img, W, H)) {
				fprintf(stderr, "Failed to write PNG.\n");
				exit(-1);
			}
		}
		free(img);
	}

	cairo_surface_destroy(target);
	cairo_destroy(cairo);

	return 0;
}
