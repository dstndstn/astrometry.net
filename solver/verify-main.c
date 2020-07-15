/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/**

 wget "http://antwrp.gsfc.nasa.gov/apod/image/0403/cmsky_cortner_full.jpg"
 solve-field --backend-config backend.cfg -v --keep-xylist %s.xy --continue --scale-low 10 --scale-units degwidth --no-tweak cmsky_cortner_full.jpg
 # index-219 ==> 159 index objs.
 cp cmsky_cortner_full.xy 1.xy
 cp cmsky_cortner_full.rdls 1.rd
 cp cmsky_cortner_full.wcs 1.wcs
 cp cmsky_cortner_full.jpg 1.jpg
 # or:
 # wget "http://live.astrometry.net/status.php?job=alpha-201003-01883980&get=wcs.fits" -O 1.wcs
 # wget "http://live.astrometry.net/status.php?job=alpha-201003-01883980&get=index.rd.fits" -O 1.rd
 verify -w 1.wcs -x 1.xy -r 1.rd -v

 **/

#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "matchobj.h"
#include "xylist.h"
#include "rdlist.h"
#include "ioutils.h"
#include "starkd.h"
#include "boilerplate.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"
#include "fitsioutils.h"
#include "fit-wcs.h"
#include "verify.h"
#include "histogram2d.h"

#include "plotstuff.h"
#include "plotimage.h"
#include "cairoutils.h"

static const char* OPTIONS = "hx:w:r:vj:p:";

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -w <WCS input file>\n"
           "   -x <xyls input file>\n"
           "   -r <rdls input file>\n"
           "   [-p <plot output file>]\n"
           "   [-v]: verbose\n"
           "   [-j <pixel-jitter>]: set pixel jitter (default 1.0)\n"
           "\n", progname);
}


int main(int argc, char** args) {
    int c;
    char* xylsfn = NULL;
    char* wcsfn = NULL;
    char* rdlsfn = NULL;
    char* plotfn = NULL;

    xylist_t* xyls = NULL;
    rdlist_t* rdls = NULL;
    sip_t sip;
    int i;
    int W, H;
    double pixeljitter = 1.0;
    int loglvl = LOG_MSG;
    double wcsscale;

    fits_use_error_system();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'p':
            plotfn = optarg;
            break;
        case 'j':
            pixeljitter = atof(optarg);
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'r':
            rdlsfn = optarg;
            break;
        case 'x':
            xylsfn = optarg;
            break;
        case 'w':
            wcsfn = optarg;
            break;
        case 'v':
            loglvl++;
            break;
        }
    }
    if (optind != argc) {
        print_help(args[0]);
        exit(-1);
    }
    if (!xylsfn || !wcsfn || !rdlsfn) {
        print_help(args[0]);
        exit(-1);
    }
    log_init(loglvl);

    // read WCS.
    logmsg("Trying to parse SIP header from %s...\n", wcsfn);
    if (!sip_read_header_file(wcsfn, &sip)) {
        logmsg("Failed to parse SIP header from %s.\n", wcsfn);
    }
    // image W, H
    W = sip.wcstan.imagew;
    H = sip.wcstan.imageh;
    if ((W == 0.0) || (H == 0.0)) {
        logmsg("WCS file %s didn't contain IMAGEW and IMAGEH headers.\n", wcsfn);
        // FIXME - use bounds of xylist?
        exit(-1);
    }
    wcsscale = sip_pixel_scale(&sip);
    logmsg("WCS scale: %g arcsec/pixel\n", wcsscale);

    // read XYLS.
    xyls = xylist_open(xylsfn);
    if (!xyls) {
        logmsg("Failed to read an xylist from file %s.\n", xylsfn);
        exit(-1);
    }

    // read RDLS.
    rdls = rdlist_open(rdlsfn);
    if (!rdls) {
        logmsg("Failed to read an rdlist from file %s.\n", rdlsfn);
        exit(-1);
    }

    {
        // (x,y) positions of field stars.
        double* fieldpix;
        int Nfield;
        double* indexpix;
        starxy_t* xy;
        rd_t* rd;
        int Nindex;

        xy = xylist_read_field(xyls, NULL);
        if (!xy) {
            logmsg("Failed to read xyls entries.\n");
            exit(-1);
        }
        Nfield = starxy_n(xy);
        fieldpix = starxy_to_xy_array(xy, NULL);
        logmsg("Found %i field objects\n", Nfield);

        // Project RDLS into pixel space.
        rd = rdlist_read_field(rdls, NULL);
        if (!rd) {
            logmsg("Failed to read rdls entries.\n");
            exit(-1);
        }
        Nindex = rd_n(rd);
        logmsg("Found %i indx objects\n", Nindex);
        indexpix = malloc(2 * Nindex * sizeof(double));
        for (i=0; i<Nindex; i++) {
            anbool ok;
            double ra = rd_getra(rd, i);
            double dec = rd_getdec(rd, i);
            ok = sip_radec2pixelxy(&sip, ra, dec, indexpix + i*2, indexpix + i*2 + 1);
            assert(ok);
        }

        logmsg("CRPIX is (%g,%g)\n", sip.wcstan.crpix[0], sip.wcstan.crpix[1]);

        {
            double* fieldsigma2s = malloc(Nfield * sizeof(double));
            int besti;
            int* theta;
            double logodds;
            double Q2, R2;
            double qc[2];
            double gamma;

            // HACK -- quad radius-squared
            Q2 = square(100.0);
            qc[0] = sip.wcstan.crpix[0];
            qc[1] = sip.wcstan.crpix[1];
            // HACK -- variance growth rate wrt radius.
            gamma = 1.0;

            for (i=0; i<Nfield; i++) {
                R2 = distsq(qc, fieldpix + 2*i, 2);
                fieldsigma2s[i] = square(pixeljitter) * (1.0 + gamma * R2/Q2);
            }

            logodds = verify_star_lists(indexpix, Nindex,
                                        fieldpix, fieldsigma2s, Nfield,
                                        W*H,
                                        0.25,
                                        log(1e-100),
                                        log(1e100),
                                        &besti, NULL, &theta, NULL);

            logmsg("Logodds: %g\n", logodds);

            if (TRUE) {
                for (i=0; i<Nfield; i++) {
                    if (theta[i] < 0)
                        continue;
                    printf("%g %g %g %g\n", fieldpix[2*i+0], fieldpix[2*i+1],
                           rd_getra(rd, theta[i]), rd_getdec(rd, theta[i]));
                }
            }

            if (plotfn) {
                plot_args_t pargs;
                plotimage_t* img;
                cairo_t* cairo;

                plotstuff_init(&pargs);
                pargs.outformat = PLOTSTUFF_FORMAT_PNG;
                pargs.outfn = plotfn;
                img = plotstuff_get_config(&pargs, "image");
                img->format = PLOTSTUFF_FORMAT_JPG;
                plot_image_set_filename(img, "1.jpg");
                plot_image_setsize(&pargs, img);
                plotstuff_run_command(&pargs, "image");
                cairo = pargs.cairo;
                // red circles around every field star.
                cairo_set_color(cairo, "red");
                for (i=0; i<Nfield; i++) {
                    cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                                           fieldpix[2*i+0], fieldpix[2*i+1],
                                           2.0 * sqrt(fieldsigma2s[i]));
                    cairo_stroke(cairo);
                }
                // green crosshairs at every index star.
                cairo_set_color(cairo, "green");
                for (i=0; i<Nindex; i++) {
                    cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_XCROSSHAIR,
                                           indexpix[2*i+0], indexpix[2*i+1],
                                           3);
                    cairo_stroke(cairo);
                }

                // thick white circles for corresponding field stars.
                cairo_set_line_width(cairo, 2);
                for (i=0; i<Nfield; i++) {
                    if (theta[i] < 0)
                        continue;
                    cairo_set_color(cairo, "white");
                    cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                                           fieldpix[2*i+0], fieldpix[2*i+1],
                                           2.0 * sqrt(fieldsigma2s[i]));
                    cairo_stroke(cairo);
                    // thick cyan crosshairs for corresponding index stars.
                    cairo_set_color(cairo, "cyan");
                    cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_XCROSSHAIR,
                                           indexpix[2*theta[i]+0],
                                           indexpix[2*theta[i]+1],
                                           3);
                    cairo_stroke(cairo);
					
                }

                plotstuff_output(&pargs);
            }

            free(theta);
            free(fieldsigma2s);
        }

        free(fieldpix);
        free(indexpix);
    }



    if (xylist_close(xyls)) {
        logmsg("Failed to close XYLS file.\n");
    }
    return 0;
}
