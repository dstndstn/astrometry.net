/*
  This file is part of the Astrometry.net suite.
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

 plot_color (<r> <g> <b> <a> or color name)
 plot_lw <linewidth>
 plot_bgcolor (<r> <g> <b> <a> or color name)
 plot_bglw <linewidth>
 plot_marker <marker-shape>
 plot_markersize <radius>
 plot_wcs <filename>
 plot_wcs_setsize
 plot_wcs_box <ra> <dec> <width>  -- center on RA,Dec with width in deg.

 Fill with a color:

 fill

 Image:

 image_file <fn>
 image_format <format>
 image_ext <extension> -- FITS extension
 image_wcs <fn>    -- project the image through its WCS, then back through the plot_wcs.
 image_setsize    -- set plot size to image size.
 image_low -- FITS pixel value that will be black.
 image_high -- FITS pixel value that will be black.
 image_null -- FITS pixel value that will be transparent.
 image

 Xy:

 Note, xylists are assumed to contain FITS-indexed pixels: the center of
 the "lower-left" pixel is at coordinate (1,1).

 xy_file <xylist>
 xy_wcs <wcs-file>
 xy_ext <fits-extension>
 xy_xcol <column-name>
 xy_ycol <column-name>
 xy_xoff <pixel-offset>  -- default 1, for FITS pixels.
 xy_yoff <pixel-offset>
 xy_firstobj <obj-num>
 xy_nobjs <n>
 xy_scale <factor>
 xy_vals <x> <y>  -- coordinates to plot
 xy

 RA,Dec lists:

 radec_file <filename (FITS binary table)>
 radec_ext <FITS extension, default 1>
 radec_racol <FITS column name, default "RA">
 radec_deccol <FITS column name, default "DEC">
 radec_firstobj <int, default 0>
 radec_nobjs <int, default no limit>
 radec_vals <ra> <dec>   -- coordinates to plot

 Annotations:

 annotations_bgcolor
 annotations_fontsize
 annotations_font
 annotations_target <ra> <dec> <label>
 annotations_targetname <label>  -- eg "M 8", "NGC 427"
 annotations_no_ngc     -- don't plot NGC/IC galaxies
 annotations_no_bright  -- don't plot named stars
 annotations_ngc_size <fraction> -- default 0.02, min size to annotate.
 annotations

 Image outlines:

 outline_wcs <fn>
 outline_step <pix> -- step size for walking the image boundary, default 10
 outline

 RA,Dec grid:

 grid_rastep <deg>
 grid_decstep <deg>
 grid_ralabelstep <deg>
 grid_declabelstep <deg>
 grid_step <deg>     -- sets all the above at once.
 grid

 Astrometry.net index files:

 index_file <filename>
 index_draw_stars <0/1>   -- draw stars?
 index_draw_quads <0/1>   -- draw quads?
 index

 Healpix boundaries:

 healpix_nside <int>
 healpix


 */

#include <stdio.h>

#include "plotstuff.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"

static const char* OPTIONS = "hvW:H:o:JjP";

static void printHelp(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] > output.png\n"
           "  [-o <output-file>] (default: stdout)\n"
           "  [-P]              Write PPM output instead of PNG.\n"
		   "  [-j]              Write JPEG output.\n"
		   "  [-J]              Write PDF output.\n"
		   "  [-W <width>   ]   Width of output image (default: data-dependent).\n"
		   "  [-H <height>  ]   Height of output image (default: data-dependent).\n"
		   "  [-v]: +verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *args[]) {
	int loglvl = LOG_MSG;
	int argchar;
	char* progname = args[0];
	plot_args_t pargs;

	plotstuff_init(&pargs);
	pargs.fout = stdout;
	pargs.outformat = PLOTSTUFF_FORMAT_PNG;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
		case 'v':
			loglvl++;
			break;
        case 'o':
            pargs.outfn = optarg;
            break;
        case 'P':
            pargs.outformat = PLOTSTUFF_FORMAT_PPM;
            break;
		case 'j':
            pargs.outformat = PLOTSTUFF_FORMAT_JPG;
			break;
		case 'J':
            pargs.outformat = PLOTSTUFF_FORMAT_PDF;
			break;
		case 'W':
			pargs.W = atoi(optarg);
			break;
		case 'H':
			pargs.H = atoi(optarg);
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
	log_init(loglvl);

    // log errors to stderr, not stdout.
    errors_log_to(stderr);

	fits_use_error_system();

	for (;;) {
		if (plotstuff_read_and_run_command(&pargs, stdin))
			break;
	}

	if (plotstuff_output(&pargs))
		exit(-1);

	plotstuff_free(&pargs);

	return 0;
}
