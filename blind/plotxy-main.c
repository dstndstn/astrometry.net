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
#include "plotimage.h"
#include "xylist.h"
#include "boilerplate.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"

#define OPTIONS "hW:H:n:N:r:s:i:e:x:y:w:S:I:PC:X:Y:b:o:pJ"

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
		   "  [-x <x-offset>]   X offset: position of the bottom-left pixel.\n"
		   "  [-y <y-offset>]   Y offset: position of the bottom-left pixel.\n"
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
    char* infn = NULL;
	char* fname = NULL;
	int n = 0, N = 0;
	double xoff = 0.0, yoff = 0.0;
	int ext = 1;
	double rad = 5.0;
	double lw = 1.0;
	char* shape = "circle";
	double scale = 1.0;
    bool pnginput = FALSE;
    char* xcol = NULL;
    char* ycol = NULL;
	unsigned char* img = NULL;

	plot_args_t pargs;
	char* fgcolor = NULL;
	char* bgcolor = NULL;

	memset(&pargs, 0, sizeof(pargs));
	pargs.fout = stdout;
	pargs.outformat = PLOTSTUFF_FORMAT_PNG;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 'C':
			fgcolor = optarg;
            break;
        case 'b':
			bgcolor = optarg;
            break;
        case 'o':
            pargs.outfn = optarg;
            break;
        case 'X':
            xcol = optarg;
            break;
        case 'Y':
            ycol = optarg;
            break;
        case 'P':
            pargs.outformat = PLOTSTUFF_FORMAT_PPM;
            break;
		case 'J':
            pargs.outformat = PLOTSTUFF_FORMAT_PDF;
			break;
		case 'p':
			pnginput = TRUE;
            break;
        case 'I':
            infn = optarg;
            break;
		case 'S':
			scale = atof(optarg);
			break;
		case 'i':
			fname = optarg;
			break;
		case 'x':
			xoff = atof(optarg);
			break;
		case 'y':
			yoff = atof(optarg);
			break;
		case 'W':
			pargs.W = atoi(optarg);
			break;
		case 'H':
			pargs.H = atoi(optarg);
			break;
		case 'n':
			n = atoi(optarg);
			break;
		case 'N':
			N = atoi(optarg);
			break;
		case 'e':
			ext = atoi(optarg);
			break;
		case 'r':
			rad = atof(optarg);
			break;
		case 'w':
			lw = atof(optarg);
			break;
		case 's':
			shape = optarg;
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
    if (infn && (pargs.W || pargs.H)) {
        printf("Error: if you specify an input file, you can't give -W or -H (width or height) arguments.\n\n");
        printHelp(progname);
        exit(-1);
    }
	if (!fname) {
		printHelp(progname);
		exit(-1);
	}

    // log errors to stderr, not stdout.
    errors_log_to(stderr);

	if (infn) {
		// HACK -- open the image file to get W,H
		if (pnginput) {
			img = cairoutils_read_png(infn, &(pargs.W), &(pargs.H));
		} else {
			logmsg("Reading PPM from \"%s\"\n", infn);
			img = cairoutils_read_ppm(infn, &(pargs.W), &(pargs.H));
		}
		if (!img) {
			ERROR("Failed to read image to get image W,H.");
			exit(-1);
		}
	}

	plotstuff_init(&pargs);

	plot_image_rgba_data(pargs.cairo, img, pargs.W, pargs.H);

	plotstuff_run_commandf(&pargs, "xy_file %s", fname);
	plotstuff_run_commandf(&pargs, "xy_ext %i", ext);
	if (xcol)
		plotstuff_run_commandf(&pargs, "xy_xcol %s", xcol);
	if (ycol)
		plotstuff_run_commandf(&pargs, "xy_xcol %s", ycol);
	if (N)
		plotstuff_run_commandf(&pargs, "xy_nobjs %i", N);
	if (n)
		plotstuff_run_commandf(&pargs, "xy_firstobj %i", n);
	plotstuff_run_commandf(&pargs, "xy_xoff %g", xoff);
	plotstuff_run_commandf(&pargs, "xy_yoff %g", yoff);

	if (fgcolor)
		plotstuff_run_commandf(&pargs, "plot_color %s", fgcolor);
	if (bgcolor)
		plotstuff_run_commandf(&pargs, "xy_bgcolor %s", bgcolor);
	plotstuff_run_commandf(&pargs, "plot_marker %s", shape);
	plotstuff_run_commandf(&pargs, "plot_markersize %g", rad);
	plotstuff_run_commandf(&pargs, "plot_lw %g", lw);
	plotstuff_run_commandf(&pargs, "xy_scale %g", scale);
	plotstuff_run_command(&pargs, "xy");

	plotstuff_output(&pargs);
	plotstuff_free(&pargs);

	return 0;
}
