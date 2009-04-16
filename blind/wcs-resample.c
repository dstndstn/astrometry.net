/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, Dustin Lang.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>
#include <assert.h>

#include "sip_qfits.h"
#include "an-bool.h"
#include "qfits.h"
#include "starutil.h"
#include "bl.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "fitsioutils.h"

const char* OPTIONS = "hw:";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s [options] <input-FITS-image> <target-WCS-file> <output-FITS-image>\n"
		   "   [-w <input WCS file>] (default is to read WCS from input FITS image)\n"
		   //"  [-t]: just use TAN projection, even if SIP extension exists.\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* inwcsfn = NULL;
	char* outwcsfn = NULL;
    char* infitsfn = NULL;
    char* outfitsfn = NULL;

    sip_t inwcs;
    sip_t outwcs;
    qfitsloader qinimg;
    qfitsdumper qoutimg;
    float* inimg;
    float* outimg;
    qfits_header* hdr;

    int outW, outH;
    int inW, inH;

	int i,j;

    double inxmin, inxmax, inymin, inymax;

    double outpixmin, outpixmax;

    //float fa;
    //int a;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
        case 'w':
            inwcsfn = optarg;
            break;
		}
	}

    log_init(LOG_MSG);
    fits_use_error_system();

	if (optind != argc - 3) {
		print_help(args[0]);
		exit(-1);
	}

    infitsfn  = args[optind+0];
    outwcsfn  = args[optind+1];
    outfitsfn = args[optind+2];

    if (!inwcsfn)
        inwcsfn = infitsfn;

	// read input WCS.
    if (!sip_read_header_file(inwcsfn, &inwcs)) {
        ERROR("Failed to parse TAN/SIP header from %s", inwcsfn);
        exit(-1);
    }

	// read output WCS.
    if (!sip_read_header_file(outwcsfn, &outwcs)) {
        ERROR("Failed to parse TAN/SIP header from %s", outwcsfn);
        exit(-1);
    }

    outW = outwcs.wcstan.imagew;
    outH = outwcs.wcstan.imageh;

    // read input image.
    memset(&qinimg, 0, sizeof(qinimg));
    qinimg.filename = infitsfn;
    // primary extension
    qinimg.xtnum = 0;
    // first pixel plane
    qinimg.pnum = 0;
    // read as floats
    qinimg.ptype = PTYPE_FLOAT;

    if (qfitsloader_init(&qinimg) ||
        qfits_loadpix(&qinimg)) {
        ERROR("Failed to read pixels from input FITS image \"%s\"", infitsfn);
        exit(-1);
    }

    // lx, ly, fbuf
    inimg = qinimg.fbuf;
    assert(inimg);
    inW = qinimg.lx;
    inH = qinimg.ly;

    logmsg("Input  image is %i x %i pixels.\n", inW, inH);
    logmsg("Output image is %i x %i pixels.\n", outW, outH);

    outimg = calloc(outW * outH, sizeof(float));

    // FIXME - the window size should depend on the relative sizes
    // of the input and output images.
    //fa = 3.0;
    //a = ceil(fa);

    inxmax = -HUGE_VAL;
    inymax = -HUGE_VAL;
    inxmin =  HUGE_VAL;
    inymin =  HUGE_VAL;

    for (j=0; j<outH; j++) {
        for (i=0; i<outW; i++) {
            double xyz[3];
            double inx, iny;
            int x,y;
            //int xlo,xhi,ylo,yhi;
            // +1 for FITS pixel coordinates.
            sip_pixelxy2xyzarr(&outwcs, i+1, j+1, xyz);
            if (!sip_xyzarr2pixelxy(&inwcs, xyz, &inx, &iny))
                continue;

            // FIXME - Nearest-neighbour resampling!!
            // -1 for FITS pixel coordinates.
            x = round(inx - 1.0);
            y = round(iny - 1.0);

            // keep track of the bounds of the requested pixels in the
            // input image.
            inxmax = MAX(inxmax, x);
            inymax = MAX(inymax, y);
            inxmin = MIN(inxmin, x);
            inymin = MIN(inymin, y);

            if (x < 0 || x >= inW || y < 0 || y >= inH)
                continue;
            outimg[j * outW + i] = inimg[y * inW + x];

            /*
             ylo = MAX(0, iny-a);
             yhi = MIN(inH-1, iny+a);
             xlo = MAX(0, inx-a);
             xhi = MIN(inW-1, inx+a);
             for (y=ylo; y<=yhi; y++) {
             for (x=xlo; x<=xhi; x++) {
             }
             }
             */
        }
    }

    logmsg("Bounds of the pixels requested from the input image:\n");
    logmsg("  x: %g to %g\n", inxmin, inxmax);
    logmsg("  y: %g to %g\n", inymin, inymax);

    {
        double pmin, pmax;
        pmin =  HUGE_VAL;
        pmax = -HUGE_VAL;
        for (i=0; i<(inW*inH); i++) {
            pmin = MIN(pmin, inimg[i]);
            pmax = MAX(pmax, inimg[i]);
        }
        logmsg("Input image bounds: %g to %g\n", pmin, pmax);
        pmin =  HUGE_VAL;
        pmax = -HUGE_VAL;
        for (i=0; i<(outW*outH); i++) {
            pmin = MIN(pmin, outimg[i]);
            pmax = MAX(pmax, outimg[i]);
        }
        logmsg("Output image bounds: %g to %g\n", pmin, pmax);
        outpixmin = pmin;
        outpixmax = pmax;
    }

    qfitsloader_free_buffers(&qinimg);

    // prepare output image.
    memset(&qoutimg, 0, sizeof(qoutimg));
    qoutimg.filename = outfitsfn;
    qoutimg.npix = outW * outH;
    qoutimg.ptype = PTYPE_FLOAT;
    qoutimg.fbuf = outimg;
    qoutimg.out_ptype = BPP_IEEE_FLOAT;

    hdr = fits_get_header_for_image(&qoutimg, outW, NULL);
    if (outwcs.a_order)
        sip_add_to_header(hdr, &outwcs);
    else
        tan_add_to_header(hdr, &(outwcs.wcstan));

    fits_header_add_double(hdr, "DATAMIN", outpixmin, "min pixel value");
    fits_header_add_double(hdr, "DATAMAX", outpixmax, "max pixel value");

	if (fits_write_header_and_image(hdr, &qoutimg)) {
        ERROR("Failed to write image to file \"%s\"", outfitsfn);
        exit(-1);
	}
    free(outimg);

	return 0;
}
