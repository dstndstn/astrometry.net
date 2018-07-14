/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "os-features.h"
#include "sip.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "mathutil.h"
#include "boilerplate.h"
#include "errors.h"

const char* OPTIONS = "he:W:H:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "\nUsage: %s [options] <wcs-file>\n"
            "  [-e <extension>]  Read from given HDU (default 0 = primary)\n"
            "  [-W <image width>] Set/override IMAGEW\n"
            "  [-H <image height>] Set/override IMAGEH\n"
            "\n", progname);
}


int main(int argc, char** args) {
    int argchar;
    char* progname = args[0];
    char** inputfiles = NULL;
    int ninputfiles = 0;
    int ext = 0;
    sip_t wcs;
    double imw=0, imh=0;
    double rac, decc;
    double det, parity, orient, orientc;
    int rah, ram, decsign, decd, decm;
    double ras, decs;
    char* units;
    double pixscale;
    double fldw, fldh;
    double ramin, ramax, decmin, decmax;
    double mxlo, mxhi, mylo, myhi;
    double dm;
    int merczoom;
    char rastr[32];
    char decstr[32];

    while ((argchar = getopt (argc, args, OPTIONS)) != -1) {
        switch (argchar) {
        case 'e':
            ext = atoi(optarg);
            break;
        case 'W':
            imw = atof(optarg);
            break;
        case 'H':
            imh = atof(optarg);
            break;
        case 'h':
        default:
            printHelp(progname);
            exit(-1);
        }
    }
    if (optind < argc) {
        ninputfiles = argc - optind;
        inputfiles = args + optind;
    }
    if (ninputfiles != 1) {
        printHelp(progname);
        exit(-1);
    }

    if (!sip_read_header_file_ext(inputfiles[0], ext, &wcs)) {
        ERROR("failed to read WCS header from file %s, extension %i", inputfiles[0], ext);
        return -1;
    }

    if (imw == 0)
        imw = wcs.wcstan.imagew;
    if (imh == 0)
        imh = wcs.wcstan.imageh;
    if ((imw == 0.0) || (imh == 0.0)) {
        ERROR("failed to find IMAGE{W,H} in WCS file");
        return -1;
    }
    // If W,H were set on the cmdline...
    if (wcs.wcstan.imagew == 0)
        wcs.wcstan.imagew = imw;
    if (wcs.wcstan.imageh == 0)
        wcs.wcstan.imageh = imh;

    printf("crpix0 %.12g\n", wcs.wcstan.crpix[0]);
    printf("crpix1 %.12g\n", wcs.wcstan.crpix[1]);
    printf("crval0 %.12g\n", wcs.wcstan.crval[0]);
    printf("crval1 %.12g\n", wcs.wcstan.crval[1]);
    printf("ra_tangent %.12g\n", wcs.wcstan.crval[0]);
    printf("dec_tangent %.12g\n", wcs.wcstan.crval[1]);
    printf("pixx_tangent %.12g\n", wcs.wcstan.crpix[0]);
    printf("pixy_tangent %.12g\n", wcs.wcstan.crpix[1]);

    printf("imagew %.12g\n", imw);
    printf("imageh %.12g\n", imh);

    printf("cd11 %.12g\n", wcs.wcstan.cd[0][0]);
    printf("cd12 %.12g\n", wcs.wcstan.cd[0][1]);
    printf("cd21 %.12g\n", wcs.wcstan.cd[1][0]);
    printf("cd22 %.12g\n", wcs.wcstan.cd[1][1]);

    det = sip_det_cd(&wcs);
    parity = (det >= 0 ? 1.0 : -1.0);
    pixscale = sip_pixel_scale(&wcs);
    printf("det %.12g\n", det);
    printf("parity %i\n", (int)parity);
    printf("pixscale %.12g\n", pixscale);

    orient = sip_get_orientation(&wcs);
    printf("orientation %.8g\n", orient);

    sip_get_radec_center(&wcs, &rac, &decc);
    printf("ra_center %.12g\n", rac);
    printf("dec_center %.12g\n", decc);

    // contributed by Rob Johnson, user rob at the domain whim.org, Nov 13, 2009
    orientc = orient + rad2deg(atan(tan(deg2rad(rac - wcs.wcstan.crval[0])) * sin(deg2rad(wcs.wcstan.crval[1]))));
    printf("orientation_center %.8g\n", orientc);

    sip_get_radec_center_hms(&wcs, &rah, &ram, &ras, &decsign, &decd, &decm, &decs);
    printf("ra_center_h %i\n", rah);
    printf("ra_center_m %i\n", ram);
    printf("ra_center_s %.12g\n", ras);
    printf("dec_center_sign %i\n", decsign);
    printf("dec_center_d %i\n", decd);
    printf("dec_center_m %i\n", decm);
    printf("dec_center_s %.12g\n", decs);

    sip_get_radec_center_hms_string(&wcs, rastr, decstr);
    printf("ra_center_hms %s\n", rastr);
    printf("dec_center_dms %s\n", decstr);

    // mercator
    printf("ra_center_merc %.8g\n", ra2mercx(rac));
    printf("dec_center_merc %.8g\n", dec2mercy(decc));

    fldw = imw * pixscale;
    fldh = imh * pixscale;
    // area of the field, in square degrees.
    printf("fieldarea %g\n", (arcsec2deg(fldw) * arcsec2deg(fldh)));

    sip_get_field_size(&wcs, &fldw, &fldh, &units);
    printf("fieldw %.4g\n", fldw);
    printf("fieldh %.4g\n", fldh);
    printf("fieldunits %s\n", units);

    sip_get_radec_bounds(&wcs, 10, &ramin, &ramax, &decmin, &decmax);
    printf("decmin %g\n", decmin);
    printf("decmax %g\n", decmax);
    printf("ramin %g\n", ramin);
    printf("ramax %g\n", ramax);

    // merc zoom level
    mxlo = ra2mercx(ramax);
    mxhi = ra2mercx(ramin);
    mylo = dec2mercy(decmax);
    myhi = dec2mercy(decmin);
    printf("ra_min_merc %g\n", mxlo);
    printf("ra_max_merc %g\n", mxhi);
    printf("dec_min_merc %g\n", mylo);
    printf("dec_max_merc %g\n", myhi);

    dm = MAX(fabs(mxlo - mxhi), fabs(mylo - myhi));
    printf("merc_diff %g\n", dm);
    merczoom = 0 - (int)floor(log(dm) / log(2.0));
    printf("merczoom %i\n", merczoom);
    return 0;
}
