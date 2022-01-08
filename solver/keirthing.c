/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>

#include "os-features.h"
#include "bl.h"
#include "fit-wcs.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"
#include "errors.h"
#include "tweak.h"
#include "mathutil.h"

static const char* OPTIONS = "hW:H:X:Y:vo:";


int main(int argc, char** args) {
    int c;
    dl* xys = dl_new(16);
    dl* radecs = dl_new(16);
    dl* otherradecs = dl_new(16);

    double* xy;
    double* xyz;
    int i, N;
    tan_t tan, tan2, tan3;
    int W=0, H=0;
    double crpix[] = { LARGE_VAL, LARGE_VAL };
    int loglvl = LOG_MSG;
    FILE* logstream = stderr;
    int order = 1;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'v':
            loglvl++;
            break;
        case 'h':
            exit(0);
        case 'o':
            order = atoi(optarg);
            break;
        case 'W':
            W = atoi(optarg);
            break;
        case 'H':
            H = atoi(optarg);
            break;
        case 'X':
            crpix[0] = atof(optarg);
            break;
        case 'Y':
            crpix[1] = atof(optarg);
            break;
        }
    }
    if (optind != argc) {
        exit(-1);
    }
    log_init(loglvl);
    log_to(logstream);
    errors_log_to(logstream);

    if (W == 0 || H == 0) {
        logerr("Need -W, -H\n");
        exit(-1);
    }
    if (crpix[0] == LARGE_VAL)
        crpix[0] = W/2.0;
    if (crpix[1] == LARGE_VAL)
        crpix[1] = H/2.0;

    while (1) {
        double x,y,ra,dec;
        if (fscanf(stdin, "%lf %lf %lf %lf\n", &x, &y, &ra, &dec) < 4)
            break;
        if (x == -1 && y == -1) {
            dl_append(otherradecs, ra);
            dl_append(otherradecs, dec);
        } else {
            dl_append(xys, x);
            dl_append(xys, y);
            dl_append(radecs, ra);
            dl_append(radecs, dec);
        }
    }
    logmsg("Read %i x,y,ra,dec tuples\n", dl_size(xys)/2);

    N = dl_size(xys)/2;
    xy = dl_to_array(xys);
    xyz = malloc(3 * N * sizeof(double));
    for (i=0; i<N; i++)
        radecdeg2xyzarr(dl_get(radecs, 2*i), dl_get(radecs, 2*i+1), xyz + i*3);
    dl_free(xys);
    dl_free(radecs);

    fit_tan_wcs(xyz, xy, N, &tan, NULL);
    tan.imagew = W;
    tan.imageh = H;

    logmsg("Computed TAN WCS:\n");
    tan_print_to(&tan, logstream);

    sip_t* sip;
    {
        tweak_t* t = tweak_new();
        starxy_t* sxy = starxy_new(N, FALSE, FALSE);
        il* imginds = il_new(256);
        il* refinds = il_new(256);

        for (i=0; i<N; i++) {
            starxy_set_x(sxy, i, xy[2*i+0]);
            starxy_set_y(sxy, i, xy[2*i+1]);
        }
        tweak_init(t);
        tweak_push_ref_xyz(t, xyz, N);
        tweak_push_image_xy(t, sxy);
        for (i=0; i<N; i++) {
            il_append(imginds, i);
            il_append(refinds, i);
        }
        // unweighted; no dist2s
        tweak_push_correspondence_indices(t, imginds, refinds, NULL, NULL);

        tweak_push_wcs_tan(t, &tan);
        t->sip->a_order = t->sip->b_order = t->sip->ap_order = t->sip->bp_order = order;

        for (i=0; i<10; i++) {
            // go to TWEAK_HAS_LINEAR_CD -> do_sip_tweak
            // t->image has the indices of corresponding image stars
            // t->ref   has the indices of corresponding catalog stars
            tweak_go_to(t, TWEAK_HAS_LINEAR_CD);
            logmsg("\n");
            sip_print(t->sip);
            t->state &= ~TWEAK_HAS_LINEAR_CD;
        }
        tan_write_to_file(&t->sip->wcstan, "kt1.wcs");
        sip = t->sip;
    }

    for (i=0; i<dl_size(otherradecs)/2; i++) {
        double ra, dec, x,y;
        ra = dl_get(otherradecs, 2*i);
        dec = dl_get(otherradecs, 2*i+1);
        if (!sip_radec2pixelxy(sip, ra, dec, &x, &y)) {
            logerr("Not in tangent plane: %g,%g\n", ra, dec);
            exit(-1);
            //continue;
        }
        printf("%g %g\n", x, y);
    }

    /*
     fit_tan_wcs_move_tangent_point(xyz, xy, N, crpix, &tan, &tan2);
     fit_tan_wcs_move_tangent_point(xyz, xy, N, crpix, &tan2, &tan3);
     logmsg("Moved tangent point to (%g,%g):\n", crpix[0], crpix[1]);
     tan_print_to(&tan3, logstream);
     tan_write_to_file(&tan, "kt1.wcs");
     tan_write_to_file(&tan3, "kt2.wcs");
     */

    dl_free(otherradecs);
    free(xy);
    free(xyz);
    return 0;
}



