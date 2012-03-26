/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009, 2012 Dustin Lang.

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
#include <assert.h>

#include <cairo.h>
#include <cairo-pdf.h>

#include "plotstuff.h"
#include "plotxy.h"
#include "plotimage.h"
#include "xylist.h"
#include "boilerplate.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"
#include "ioutils.h"

#define OPTIONS "hvW:H:n:N:r:s:i:e:x:y:w:S:I:PC:X:Y:b:o:pJ"

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] > output.png\n"
		   "  -i <input-file>   Input file (xylist)\n"
           "  [-o <output-file>] (default: stdout)\n"
           "  [-I <image>   ]   Input image on which plotting will occur; PPM format.\n"
	       "  [-p]: Input image is PNG format, not PPM.\n"
           "  [-P]              Write PPM output instead of PNG.\n"
		   "  [-J]              Write PDF output.\n"
		   "  [-W <width>   ]   Width of output image (default: data-dependent).\n"
		   "  [-H <height>  ]   Height of output image (default: data-dependent).\n"
		   "  [-x <x-offset>]   X offset: position of the bottom-left pixel (default: 1).\n"
		   "  [-y <y-offset>]   Y offset: position of the bottom-left pixel (default: 1).\n"
		   "  [-X <x-column-name>] X column: name of the FITS column.\n"
		   "  [-Y <y-column-name>] Y column: name of the FITS column.\n"
		   "  [-n <first-obj>]  First object to plot (default: 0).\n"
		   "  [-N <num-objs>]   Number of objects to plot (default: all).\n"
		   "  [-r <radius>]     Size of markers to plot (default: 5.0).\n"
		   "  [-w <linewidth>]  Linewidth (default: 1.0).\n"
		   "  [-s <shape>]      Shape of markers (default: circle):",
           progname);
    cairoutils_print_marker_names("\n                 ");
    printf("\n");
    printf("  [-C <color>]      Color to plot in: (default: white)\n");
    cairoutils_print_color_names("\n                 ");
    printf("\n");
    printf("  [-b <color>]      Draw in <color> behind each marker.\n"
           "  [-S <scale-factor>]  Scale xylist entries by this value before plotting.\n"
		   "  [-e <extension>]  FITS extension to read (default 0).\n"
		   "\n");
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *args[]) {
	int argchar;
	char* progname = args[0];

	plot_args_t pargs;
	plotxy_t* xy;
	plotimage_t* img;
	int loglvl = LOG_MSG;

    // log errors to stderr, not stdout.
    errors_log_to(stderr);

	plotstuff_init(&pargs);
	pargs.fout = stdout;
	pargs.outformat = PLOTSTUFF_FORMAT_PNG;

	xy = plotstuff_get_config(&pargs, "xy");
	img = plotstuff_get_config(&pargs, "image");
	assert(xy);
	assert(img);

	plotstuff_set_color(&pargs, "white");
	plotstuff_set_bgcolor(&pargs, "black");
	
	img->format = PLOTSTUFF_FORMAT_PPM;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
		case 'v':
			loglvl++;
			break;
        case 'C':
			plotstuff_set_color(&pargs, optarg);
            break;
        case 'b':
			plotstuff_set_bgcolor(&pargs, "optarg");
            break;
        case 'o':
            pargs.outfn = optarg;
            break;
        case 'X':
			plot_xy_set_xcol(xy, optarg);
            break;
        case 'Y':
			plot_xy_set_ycol(xy, optarg);
            break;
        case 'P':
            pargs.outformat = PLOTSTUFF_FORMAT_PPM;
            break;
		case 'J':
            pargs.outformat = PLOTSTUFF_FORMAT_PDF;
			break;
		case 'p':
			img->format = PLOTSTUFF_FORMAT_PNG;
            break;
        case 'I':
			plot_image_set_filename(img, optarg);
            break;
		case 'S':
			xy->scale = atof(optarg);
			break;
		case 'i':
			plot_xy_set_filename(xy, optarg);
			break;
		case 'x':
			xy->xoff = atof(optarg);
			break;
		case 'y':
			xy->yoff = atof(optarg);
			break;
		case 'W':
			pargs.W = atoi(optarg);
			break;
		case 'H':
			pargs.H = atoi(optarg);
			break;
		case 'n':
			xy->firstobj = atoi(optarg);
			break;
		case 'N':
			xy->nobjs = atoi(optarg);
			break;
		case 'e':
			xy->ext = atoi(optarg);
			break;
		case 'r':
			pargs.markersize = atof(optarg);
			break;
		case 'w':
			pargs.lw = atof(optarg);
			break;
		case 's':
			plotstuff_set_marker(&pargs, optarg);
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
	if (!xy->fn) {
		printHelp(progname);
		exit(-1);
	}
	log_init(loglvl);
	log_to(stderr);
	fits_use_error_system();
	if (img->fn) {
		if (plot_image_setsize(&pargs, img)) {
			ERROR("Failed to set plot size from image");
			exit(-1);
		}
		plotstuff_run_command(&pargs, "image");
	} else {
		if (pargs.W == 0 || pargs.H == 0) {
			if (plot_xy_setsize(&pargs, xy)) {
				ERROR("Failed to set plot size from xylist");
				exit(-1);
			}
		}
	}

	plotstuff_run_command(&pargs, "xy");

	plotstuff_output(&pargs);
	plotstuff_free(&pargs);

	return 0;
}
