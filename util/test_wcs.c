/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include "sip.h"
#include "sip_qfits.h"
#include "anwcs.h"
#include "fitsioutils.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "cutest.h"

const char* tan1 = "SIMPLE  =                    T / Standard FITS file                             BITPIX  =                    8 / ASCII or bytes array                           NAXIS   =                    0 / Minimal header                                 EXTEND  =                    T / There may be FITS ext                          CTYPE1  = 'RA---TAN'     / TAN projection                                       CTYPE2  = 'DEC--TAN'                                 / TAN projection           WCSAXES =                    2 / no comment                                     EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         LONPOLE =                180.0 / no comment                                     LATPOLE =                  0.0 / no comment                                     CRVAL1  =        6.71529334958 / RA  of reference point                         CRVAL2  =       -72.7547910844 / DEC of reference point                         CRPIX1  =        477.760482899 / X reference pixel                              CRPIX2  =        361.955063329 / Y reference pixel                              CUNIT1  = 'deg     ' / X pixel scale units                                      CUNIT2  = 'deg     ' / Y pixel scale units                                      CD1_1   =   -3.77685217356E-05 / Transformation matrix                          CD1_2   =    -0.00273782095286 / no comment                                     CD2_1   =     0.00273550077464 / no comment                                     CD2_2   =     -3.654231597E-05 / no comment                                     IMAGEW  =                 1024 / Image width,  in pixels.                       IMAGEH  =                  683 / Image height, in pixels.                       ";

const char* tan2 = "SIMPLE  =                    T / Standard FITS file                             BITPIX  =                    8 / ASCII or bytes array                           NAXIS   =                    0 / Minimal header                                 EXTEND  =                    T / There may be FITS ext                          CTYPE1  = 'RA---TAN-SIP' / TAN (gnomic) projection + SIP distortions            CTYPE2  = 'DEC--TAN-SIP' / TAN (gnomic) projection + SIP distortions            WCSAXES =                    2 / no comment                                     EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         LONPOLE =                180.0 / no comment                                     LATPOLE =                  0.0 / no comment                                     CRVAL1  =        6.71529334958 / RA  of reference point                         CRVAL2  =       -72.7547910844 / DEC of reference point                         CRPIX1  =        477.760482899 / X reference pixel                              CRPIX2  =        361.955063329 / Y reference pixel                              CUNIT1  = 'deg     ' / X pixel scale units                                      CUNIT2  = 'deg     ' / Y pixel scale units                                      CD1_1   =   -3.77685217356E-05 / Transformation matrix                          CD1_2   =    -0.00273782095286 / no comment                                     CD2_1   =     0.00273550077464 / no comment                                     CD2_2   =     -3.654231597E-05 / no comment                                     IMAGEW  =                 1024 / Image width,  in pixels.                       IMAGEH  =                  683 / Image height, in pixels.                       A_ORDER =                    2 / Polynomial order, axis 1                       A_0_2   =    2.25528052918E-06 / no comment                                     A_1_1   =   -4.81359073875E-07 / no comment                                     A_2_0   =    4.30780269448E-07 / no comment                                     B_ORDER =                    2 / Polynomial order, axis 2                       B_0_2   =   -1.03727867632E-06 / no comment                                     B_1_1   =     6.9796860706E-07 / no comment                                     B_2_0   =    -3.7809448368E-07 / no comment                                     AP_ORDER=                    2 / Inv polynomial order, axis 1                   AP_0_1  =   -6.65805505351E-07 / no comment                                     AP_0_2  =    -2.2549438026E-06 / no comment                                     AP_1_0  =    3.30484183954E-07 / no comment                                     AP_1_1  =    4.80936510428E-07 / no comment                                     AP_2_0  =   -4.30764936375E-07 / no comment                                     BP_ORDER=                    2 / Inv polynomial order, axis 2                   BP_0_1  =    4.50013020053E-07 / no comment                                     BP_0_2  =    1.03706596388E-06 / no comment                                     BP_1_0  =   -2.70330536785E-07 / no comment                                     BP_1_1  =   -6.97662208285E-07 / no comment                                     BP_2_0  =    3.78065361127E-07 / no comment                                     ";

const char* sin1 = "SIMPLE  =                    T / Standard FITS file                             BITPIX  =                    8 / ASCII or bytes array                           NAXIS   =                    0 / Minimal header                                 EXTEND  =                    T / There may be FITS ext                          CTYPE1  = 'RA---SIN'     / SIN projection                                       CTYPE2  = 'DEC--SIN'                                 / SIN projection           WCSAXES =                    2 / no comment                                     EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         LONPOLE =                180.0 / no comment                                     LATPOLE =                  0.0 / no comment                                     CRVAL1  =        6.71529334958 / RA  of reference point                         CRVAL2  =       -72.7547910844 / DEC of reference point                         CRPIX1  =        477.760482899 / X reference pixel                              CRPIX2  =        361.955063329 / Y reference pixel                              CUNIT1  = 'deg     ' / X pixel scale units                                      CUNIT2  = 'deg     ' / Y pixel scale units                                      CD1_1   =   -3.77685217356E-05 / Transformation matrix                          CD1_2   =    -0.00273782095286 / no comment                                     CD2_1   =     0.00273550077464 / no comment                                     CD2_2   =     -3.654231597E-05 / no comment                                     IMAGEW  =                 1024 / Image width,  in pixels.                       IMAGEH  =                  683 / Image height, in pixels.                       ";

const char* sin2 = "SIMPLE  =                    T / Standard FITS file                             BITPIX  =                    8 / ASCII or bytes array                           NAXIS   =                    0 / Minimal header                                 EXTEND  =                    T / There may be FITS ext                          CTYPE1  = 'RA---SIN-SIP' / TAN (gnomic) projection + SIP distortions            CTYPE2  = 'DEC--SIN-SIP' / TAN (gnomic) projection + SIP distortions            WCSAXES =                    2 / no comment                                     EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         LONPOLE =                180.0 / no comment                                     LATPOLE =                  0.0 / no comment                                     CRVAL1  =        6.71529334958 / RA  of reference point                         CRVAL2  =       -72.7547910844 / DEC of reference point                         CRPIX1  =        477.760482899 / X reference pixel                              CRPIX2  =        361.955063329 / Y reference pixel                              CUNIT1  = 'deg     ' / X pixel scale units                                      CUNIT2  = 'deg     ' / Y pixel scale units                                      CD1_1   =   -3.77685217356E-05 / Transformation matrix                          CD1_2   =    -0.00273782095286 / no comment                                     CD2_1   =     0.00273550077464 / no comment                                     CD2_2   =     -3.654231597E-05 / no comment                                     IMAGEW  =                 1024 / Image width,  in pixels.                       IMAGEH  =                  683 / Image height, in pixels.                       A_ORDER =                    2 / Polynomial order, axis 1                       A_0_2   =    2.25528052918E-06 / no comment                                     A_1_1   =   -4.81359073875E-07 / no comment                                     A_2_0   =    4.30780269448E-07 / no comment                                     B_ORDER =                    2 / Polynomial order, axis 2                       B_0_2   =   -1.03727867632E-06 / no comment                                     B_1_1   =     6.9796860706E-07 / no comment                                     B_2_0   =    -3.7809448368E-07 / no comment                                     AP_ORDER=                    2 / Inv polynomial order, axis 1                   AP_0_1  =   -6.65805505351E-07 / no comment                                     AP_0_2  =    -2.2549438026E-06 / no comment                                     AP_1_0  =    3.30484183954E-07 / no comment                                     AP_1_1  =    4.80936510428E-07 / no comment                                     AP_2_0  =   -4.30764936375E-07 / no comment                                     BP_ORDER=                    2 / Inv polynomial order, axis 2                   BP_0_1  =    4.50013020053E-07 / no comment                                     BP_0_2  =    1.03706596388E-06 / no comment                                     BP_1_0  =   -2.70330536785E-07 / no comment                                     BP_1_1  =   -6.97662208285E-07 / no comment                                     BP_2_0  =    3.78065361127E-07 / no comment                                     ";

static void parse(const char* hdr, sip_t** sip, anwcs_t** anwcs, CuTest* tc) {
    int len;
    int hlen;
    char* str;

    fits_use_error_system();

    len = strlen(hdr);
    hlen = fits_bytes_needed(len + 80);
    str = malloc(hlen + 1);
    memcpy(str, hdr, len);
    memset(str + len, ' ', hlen - len);
    memcpy(str + hlen - 80, "END", 3);
    str[hlen] = '\0';

    //printf("Passing FITS header string: len %i, >>\n%s\n<<\n", hlen, str);

    *sip = sip_from_string(str, hlen, NULL);
    CuAssertPtrNotNull(tc, *sip);

    *anwcs = anwcs_wcslib_from_string(str, hlen);
    CuAssertPtrNotNull(tc, *anwcs);

    free(str);
}

static void tst_rd2xy(double ra, double dec, double xtrue, double ytrue,
                      anwcs_t* an1, sip_t* sip1, tan_t* tan1, CuTest* tc) {
    double x, y;
    anbool ok;
    int err;

    if (an1) {
        x = y = 0.0;
        err = anwcs_radec2pixelxy(an1, ra, dec, &x, &y);
        printf("RA,Dec (%g, %g) -> x,y (%g, %g)\n", ra, dec, x, y);
        CuAssertIntEquals(tc, 0, err);
        CuAssertDblEquals(tc, xtrue, x, 1e-8);
        CuAssertDblEquals(tc, ytrue, y, 1e-8);
    }

    if (sip1) {
        x = y = 0.0;
        ok = sip_radec2pixelxy(sip1, ra, dec, &x, &y);
        printf("RA,Dec (%g, %g) -> x,y (%g, %g)\n", ra, dec, x, y);
        CuAssertIntEquals(tc, TRUE, ok);
        CuAssertDblEquals(tc, xtrue, x, 1e-8);
        CuAssertDblEquals(tc, ytrue, y, 1e-8);
    }

    if (tan1) {
        x = y = 0.0;
        ok = tan_radec2pixelxy(tan1, ra, dec, &x, &y);
        CuAssertIntEquals(tc, TRUE, ok);
        CuAssertDblEquals(tc, xtrue, x, 1e-8);
        CuAssertDblEquals(tc, ytrue, y, 1e-8);
    }
}

static void tst_xy2rd(double x, double y, double ratrue, double dectrue,
                      anwcs_t* an1, sip_t* sip1, tan_t* tan1, CuTest* tc) {
    double ra, dec;
    int err;

    if (an1) {
        ra = dec = 0.0;
        err = anwcs_pixelxy2radec(an1, x, y, &ra, &dec);
        printf("x,y (%g, %g) -> RA,Dec (%g, %g)\n", x, y, ra, dec);
        CuAssertIntEquals(tc, 0, err);
        CuAssertDblEquals(tc, ratrue,  ra,  1e-8);
        CuAssertDblEquals(tc, dectrue, dec, 1e-8);
    }

    if (sip1) {
        ra = dec = 0.0;
        sip_pixelxy2radec(sip1, x, y, &ra, &dec);
        printf("x,y (%g, %g) -> RA,Dec (%g, %g)\n", x, y, ra, dec);
        CuAssertDblEquals(tc, ratrue,  ra,  1e-8);
        CuAssertDblEquals(tc, dectrue, dec, 1e-8);
    }

    if (tan1) {
        ra = dec = 0.0;
        tan_pixelxy2radec(tan1, x, y, &ra, &dec);
        printf("x,y (%g, %g) -> RA,Dec (%g, %g)\n", x, y, ra, dec);
        CuAssertDblEquals(tc, ratrue,  ra,  1e-8);
        CuAssertDblEquals(tc, dectrue, dec, 1e-8);
    }
}

void test_tan1(CuTest* tc) {
    sip_t   *sip1, *sip2, *sip3, *sip4;
    anwcs_t * an1, * an2, * an3, * an4;

    parse(tan1, &sip1, &an1, tc);
    parse(tan2, &sip2, &an2, tc);
    parse(sin1, &sip3, &an3, tc);
    parse(sin2, &sip4, &an4, tc);

    CuAssertIntEquals(tc, FALSE, sip1->wcstan.sin);
    CuAssertIntEquals(tc, TRUE,  sip3->wcstan.sin);

    //anwcs_print(an1, stdout);
    //sip_print(sip1);

    /*
     > wcs-rd2xy -w tan1.wcs -r 10.4 -d -74.1 
     RA,Dec (10.4000000000, -74.1000000000) -> pixel (-30.3363662787, 0.3458823213)
     */

    tst_rd2xy(10.4, -74.1, -30.3363662787, 0.3458823213,
              an1, sip1, &(sip1->wcstan), tc);

    /*
     > wcs-xy2rd -w ../tan1.wcs -x 1 -y 1
     Pixel (1.0000000000, 1.0000000000) -> RA,Dec (10.3701345009, -74.0147061244)
     */
    tst_xy2rd(1.0, 1.0, 10.3701345009, -74.0147061244,
              an1, sip1, &(sip1->wcstan), tc);


    /*
     > wcs-rd2xy -w ../sin1.wcs -r 10.4 -d -74.1 -L
     RA,Dec (10.4000000000, -74.1000000000) -> pixel (-30.1110266970, 0.5062550173)
     */

    tst_rd2xy(10.4, -74.1, -30.1110266970, 0.5062550173,
              an3, sip3, &(sip3->wcstan), tc);

    /*
     > wcs-xy2rd -w ../sin1.wcs -x 1 -y 1 -L
     Pixel (1.0000000000, 1.0000000000) -> RA,Dec (10.3717392951, -74.0152067481)
     */

    tst_xy2rd(1.0, 1.0, 10.3717392951, -74.0152067481,
              an3, sip3, &(sip3->wcstan), tc);

    // TAN-SIP / SIN-SIP

    /*
     > wcs-xy2rd -w ../tan2.wcs -x 1 -y 1
     Pixel (1.0000000000, 1.0000000000) -> RA,Dec (10.3709060366, -74.0138433058)
     */

    tst_xy2rd(1.0, 1.0, 10.3709060366, -74.0138433058,
              NULL, sip2, NULL, tc);


    /*
     > wcs-rd2xy -w tan2.wcs -r 10.4 -d -74.1
     RA,Dec (10.4000000000, -74.1000000000) -> pixel (-30.6539962452, 0.4508839887)
     */
    tst_rd2xy(10.4, -74.1, -30.6539962452, 0.4508839887,
              NULL, sip2, NULL, tc);


    /*
     > wcs-xy2rd -w sin2.wcs -x 1 -y 1
     Pixel (1.0000000000, 1.0000000000) -> RA,Dec (10.3725100913, -74.0143432622)
     */

    tst_xy2rd(1.0, 1.0, 10.3725100913, -74.0143432622,
              NULL, sip4, NULL, tc);

    /*
     > wcs-rd2xy -w sin2.wcs -r 10.4 -d -74.1
     RA,Dec (10.4000000000, -74.1000000000) -> pixel (-30.4283749577, 0.6111635583)
     */

    tst_rd2xy(10.4, -74.1, -30.4283749577, 0.6111635583,
              NULL, sip4, NULL, tc);


    // SIP round-trips

    double x, y, ra, dec, x2, y2;
    anbool ok;

    x = y = 100.0;

    ra = dec = 0.0;
    sip_pixelxy2radec(sip2, x, y, &ra, &dec);
    printf("x,y (%g, %g) -> RA,Dec (%g, %g)\n", x, y, ra, dec);

    x2 = y2 = 0.0;
    ok = sip_radec2pixelxy(sip2, ra, dec, &x2, &y2);
    printf("RA,Dec (%g, %g) -> x,y (%g, %g)\n", ra, dec, x2, y2);
    CuAssertIntEquals(tc, TRUE, ok);

    CuAssertDblEquals(tc, x, x2, 1e-3);
    CuAssertDblEquals(tc, y, y2, 1e-3);


    ra = dec = 0.0;
    sip_pixelxy2radec(sip4, x, y, &ra, &dec);
    printf("x,y (%g, %g) -> RA,Dec (%g, %g)\n", x, y, ra, dec);
    x2 = y2 = 0.0;
    ok = sip_radec2pixelxy(sip4, ra, dec, &x2, &y2);
    printf("RA,Dec (%g, %g) -> x,y (%g, %g)\n", ra, dec, x2, y2);
    CuAssertIntEquals(tc, TRUE, ok);
    CuAssertDblEquals(tc, x, x2, 1e-3);
    CuAssertDblEquals(tc, y, y2, 1e-3);

}

void test_northpole(CuTest* tc) {
    tan_t  thewcs;
    tan_t   *wcs;
    double x, y, ra, dec;

    //double dec_test = 89.7;
    double dec_test = -89.7;
    
    wcs = &thewcs;
    thewcs.crval[0] = 180.;
    //thewcs.crval[1] = +90.;
    thewcs.crval[1] = -90.;
    thewcs.crpix[0] = 1024.5;
    thewcs.crpix[1] = 1024.5;
    thewcs.cd[0][0] = -0.00152778;
    thewcs.cd[0][1] =  0.;
    thewcs.cd[1][0] =  0.;
    thewcs.cd[1][1] =  0.00152778;
    thewcs.imagew   =  2048;
    thewcs.imageh   =  2048;
    thewcs.sin      = FALSE;

    printf("\npixelxy2iwc\n");
    ra = dec = 0.0;
    tan_pixelxy2iwc(wcs, 800., 1024., &ra, &dec);
    printf("800,1024 -> %.3f, %.3f\n", ra, dec);
    ra = dec = 0.0;
    tan_pixelxy2iwc(wcs, 1024., 800., &ra, &dec);
    printf("1024,800 -> %.3f, %.3f\n", ra, dec);
    ra = dec = 0.0;
    tan_pixelxy2iwc(wcs, 1200., 1024., &ra, &dec);
    printf("1200,1024 -> %.3f, %.3f\n", ra, dec);
    ra = dec = 0.0;
    tan_pixelxy2iwc(wcs, 1024., 1200., &ra, &dec);
    printf("1024,1200 -> %.3f, %.3f\n", ra, dec);

    printf("\niwc2pixelxy\n");
    x = y = 0.0;
    tan_iwc2pixelxy(wcs, +0.3, 0., &x, &y);
    printf("+,0 -> %.1f, %.1f\n", x, y);
    x = y = 0.0;
    tan_iwc2pixelxy(wcs, 0., -0.3, &x, &y);
    printf("0,- -> %.1f, %.1f\n", x, y);
    x = y = 0.0;
    tan_iwc2pixelxy(wcs, -0.3, 0., &x, &y);
    printf("-,0 -> %.1f, %.1f\n", x, y);
    x = y = 0.0;
    tan_iwc2pixelxy(wcs, 0., +0.3, &x, &y);
    printf("0,+ -> %.1f, %.1f\n", x, y);

    printf("\niwc2radec\n");
    x = y = 0.0;
    tan_iwc2radec(wcs, +0.3, 0., &ra, &dec);
    printf("+,0 -> %.1f, %.1f\n", ra, dec);
    x = y = 0.0;
    tan_iwc2radec(wcs, 0., -0.3, &ra, &dec);
    printf("0,- -> %.1f, %.1f\n", ra, dec);
    x = y = 0.0;
    tan_iwc2radec(wcs, -0.3, 0., &ra, &dec);
    printf("-,0 -> %.1f, %.1f\n", ra, dec);
    x = y = 0.0;
    tan_iwc2radec(wcs, 0., +0.3, &ra, &dec);
    printf("0,+ -> %.1f, %.1f\n", ra, dec);

    printf("\nradec2iwc\n");
    x = y = 0.0;
    tan_radec2iwc(wcs, 0., dec_test, &x, &y);
    printf("0, %.1f -> %.3f, %.3f\n", dec_test, x, y);
    x = y = 0.0;
    tan_radec2iwc(wcs, 90., dec_test, &x, &y);
    printf("90, %.1f -> %.3f, %.3f\n", dec_test, x, y);
    x = y = 0.0;
    tan_radec2iwc(wcs, 180., dec_test, &x, &y);
    printf("180, %.1f -> %.3f, %.3f\n", dec_test, x, y);
    x = y = 0.0;
    tan_radec2iwc(wcs, 270., dec_test, &x, &y);
    printf("270, %.1f -> %.3f, %.3f\n", dec_test, x, y);
    
    
    printf("\npixelxy2radec\n");
    ra = dec = 0.0;
    tan_pixelxy2radec(wcs, 800., 1024., &ra, &dec);
    printf("800,1024 -> %.3f, %.3f\n", ra, dec);
    ra = dec = 0.0;
    tan_pixelxy2radec(wcs, 1024., 800., &ra, &dec);
    printf("1024,800 -> %.3f, %.3f\n", ra, dec);
    ra = dec = 0.0;
    tan_pixelxy2radec(wcs, 1200., 1024., &ra, &dec);
    printf("1200,1024 -> %.3f, %.3f\n", ra, dec);
    ra = dec = 0.0;
    tan_pixelxy2radec(wcs, 1024., 1200., &ra, &dec);
    printf("1024,1200 -> %.3f, %.3f\n", ra, dec);

    printf("\nradec2pixelxy\n");
    x = y = 0.0;
    tan_radec2pixelxy(wcs, 0., dec_test, &x, &y);
    printf("0, %.1f -> %.1f, %.1f\n", dec_test, x, y);
    x = y = 0.0;
    tan_radec2pixelxy(wcs, 90., dec_test, &x, &y);
    printf("90, %.1f -> %.1f, %.1f\n", dec_test, x, y);
    x = y = 0.0;
    tan_radec2pixelxy(wcs, 180., dec_test, &x, &y);
    printf("180, %.1f -> %.1f, %.1f\n", dec_test, x, y);
    x = y = 0.0;
    tan_radec2pixelxy(wcs, 270., dec_test, &x, &y);
    printf("270, %.1f -> %.1f, %.1f\n", dec_test, x, y);


    /*
     printf("\n\n\ncrval[0] = 0.\n");
     thewcs.crval[0] = 0.;
     printf("\niwc2radec\n");
     x = y = 0.0;
     tan_iwc2radec(wcs, +0.3, 0., &ra, &dec);
     printf("+,0 -> %.1f, %.1f\n", ra, dec);
     x = y = 0.0;
     tan_iwc2radec(wcs, 0., -0.3, &ra, &dec);
     printf("0,- -> %.1f, %.1f\n", ra, dec);
     x = y = 0.0;
     tan_iwc2radec(wcs, -0.3, 0., &ra, &dec);
     printf("-,0 -> %.1f, %.1f\n", ra, dec);
     x = y = 0.0;
     tan_iwc2radec(wcs, 0., +0.3, &ra, &dec);
     printf("0,+ -> %.1f, %.1f\n", ra, dec);
     */
    
}

