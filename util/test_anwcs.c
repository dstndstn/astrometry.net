/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "anwcs.h"
#include "sip.h"
#include "sip_qfits.h"
#include "errors.h"
#include "ioutils.h"
#include "log.h"

struct walk_token2 {
    dl* radecs;
};

void test_mercator_2(CuTest* tc) {
    double ra,dec;
    double ra2,dec2;
    anwcs_t* wcs = anwcs_create_mercator_2(180., 0., 128.5, 128.5,
                                           1., 256, 256, TRUE);
    printf("RA,Dec in corners:\n");
    anwcs_pixelxy2radec(wcs, 1., 1., &ra, &dec);
    CuAssertDblEquals(tc,  85.0, dec, 0.1);
    CuAssertDblEquals(tc, 360.0, ra,  1.0);
    printf("  1,1 -> %.1f, %.1f\n", ra, dec);
    anwcs_pixelxy2radec(wcs, 256., 256., &ra, &dec);
    printf("  W,H -> %.1f, %.1f\n", ra, dec);
    CuAssertDblEquals(tc, -85.0, dec, 0.1);
    CuAssertDblEquals(tc, 0.0, ra,  1.0);

    printf("Delta near the center:\n");
    anwcs_pixelxy2radec(wcs, 128., 128., &ra, &dec);
    anwcs_pixelxy2radec(wcs, 129., 128., &ra2, &dec2);
    printf("dx = 1 -> dRA, dDec %.1f, %.1f\n", ra2-ra, dec2-dec);
    anwcs_pixelxy2radec(wcs, 128., 129., &ra2, &dec2);
    printf("dy = 1 -> dRA, dDec %.1f, %.1f\n", ra2-ra, dec2-dec);
}


/*
 static void walk_callback2(const anwcs_t* wcs, double ix, double iy,
 double ra, double dec, void* token) {
 struct walk_token2* walk = token;
 dl_append(walk->radecs, ra);
 dl_append(walk->radecs, dec);
 }
 */
void test_walk_outline(CuTest* tc) {
    anwcs_t* plotwcs = anwcs_create_allsky_hammer_aitoff(180., 0., 400, 200);
#if 0
    tan_t tanwcs;
    tanwcs.crval[0] = 10.;
    tanwcs.crval[1] =  0.;
    tanwcs.crpix[0] = 25.;
    tanwcs.crpix[1] = 25.;
    tanwcs.cd[0][0] = 1.;
    tanwcs.cd[0][1] = 0.;
    tanwcs.cd[1][0] = 0.;
    tanwcs.cd[1][1] = 1.;
    tanwcs.imagew = 50.;
    tanwcs.imageh = 50.;
    tanwcs.sin = 0;
#endif

    dl* rd;
    /*
     anwcs_t* wcs = anwcs_new_tan(&tanwcs);
     struct walk_token2 token2;
     double stepsize = 1000;
     */

    log_init(LOG_ALL);

    /*
     token2.radecs = dl_new(256);
     anwcs_walk_image_boundary(wcs, stepsize, walk_callback2, &token2);
     rd = token2.radecs;
     logverb("Outline: walked in %i steps\n", dl_size(rd)/2);
     // close
     dl_append(rd, dl_get(rd, 0));
     dl_append(rd, dl_get(rd, 1));
     int i;
     for (i=0; i<dl_size(rd)/2; i++) {
     double ra,dec;
     ra  = dl_get(rd, 2*i+0);
     dec = dl_get(rd, 2*i+1);
     printf("outline step %i: (%.2f, %.2f)\n", i, ra, dec);
     }
     */

    rd = dl_new(256);
    dl_append(rd, 10.);
    dl_append(rd,  0.);

    dl_append(rd, 350.);
    dl_append(rd,  0.);

    dl_append(rd, 350.);
    dl_append(rd,  10.);

    dl_append(rd,  10.);
    dl_append(rd,  10.);

    dl_append(rd,  10.);
    dl_append(rd,   0.);

    pl* lists = anwcs_walk_outline(plotwcs, rd, 100);

    printf("\n\n\n");

    printf("%zu lists\n", pl_size(lists));
    int i;
    for (i=0; i<pl_size(lists); i++) {
        dl* plotlist = pl_get(lists, i);
        printf("List %i: length %zu\n", i, dl_size(plotlist));
        int j;
        for (j=0; j<dl_size(plotlist)/2; j++) {
            printf("  %.2f, %.2f\n",
                   dl_get(plotlist, j*2+0),
                   dl_get(plotlist, j*2+1));
        }
        printf("\n");
    }

}

void test_discontinuity(CuTest* tc) {
    double ra3,dec3,ra4,dec4;
    anbool isit;
    anwcs_t* wcs = anwcs_create_allsky_hammer_aitoff(0., 0., 400, 200);
    isit = anwcs_find_discontinuity(wcs, 3., 15., 346, 15.,
                                    &ra3, &dec3, &ra4, &dec4);
    CuAssertIntEquals(tc, 0, isit);

    isit = anwcs_find_discontinuity(wcs, 170., 0., 190., 0.,
                                    &ra3, &dec3, &ra4, &dec4);
    CuAssertIntEquals(tc, 1, isit);
    printf("RA,Dec 3: %g,%g; RA,Dec 4: %g,%g\n", ra3,dec3,ra4,dec4);
    CuAssertDblEquals(tc, 180.,  ra3, 1e-6);
    CuAssertDblEquals(tc,   0., dec3, 1e-6);
    CuAssertDblEquals(tc,-180.,  ra4, 1e-6);
    CuAssertDblEquals(tc,   0., dec4, 1e-6);

    isit = anwcs_find_discontinuity(wcs, 190., 0., 170., 0.,
                                    &ra3, &dec3, &ra4, &dec4);
    CuAssertIntEquals(tc, 1, isit);
    printf("RA,Dec 3: %g,%g; RA,Dec 4: %g,%g\n", ra3,dec3,ra4,dec4);
    CuAssertDblEquals(tc,-180.,  ra3, 1e-6);
    CuAssertDblEquals(tc,   0., dec3, 1e-6);
    CuAssertDblEquals(tc, 180.,  ra4, 1e-6);
    CuAssertDblEquals(tc,   0., dec4, 1e-6);

    anwcs_free(wcs);

    wcs = anwcs_create_allsky_hammer_aitoff(90., 0., 400, 200);

    isit = anwcs_find_discontinuity(wcs, 275., 0., 265., 0.,
                                    &ra3, &dec3, &ra4, &dec4);
    CuAssertIntEquals(tc, 1, isit);
    printf("RA,Dec 3: %g,%g; RA,Dec 4: %g,%g\n", ra3,dec3,ra4,dec4);
    CuAssertDblEquals(tc, -90.,  ra3, 1e-6);
    CuAssertDblEquals(tc,   0., dec3, 1e-6);
    CuAssertDblEquals(tc, 270.,  ra4, 1e-6);
    CuAssertDblEquals(tc,   0., dec4, 1e-6);

    anwcs_free(wcs);



    wcs = anwcs_create_allsky_hammer_aitoff(180., 0., 400, 200);

    isit = anwcs_find_discontinuity(wcs, 10., 0., 350., 0.,
                                    &ra3, &dec3, &ra4, &dec4);
    CuAssertIntEquals(tc, 1, isit);
    printf("RA,Dec 3: %g,%g; RA,Dec 4: %g,%g\n", ra3,dec3,ra4,dec4);
    CuAssertDblEquals(tc,-360.,  ra3, 1e-6);
    CuAssertDblEquals(tc,   0., dec3, 1e-6);
    CuAssertDblEquals(tc,-360.,  ra4, 1e-6);
    CuAssertDblEquals(tc,   0., dec4, 1e-6);

    anwcs_free(wcs);

}

void test_wcslib_equals_tan(CuTest* tc) {
#ifndef WCSLIB_EXISTS
    printf("\n\nWarning, WCSLIB support was not compiled in, so no WCSLIB functionality will be tested.\n\n\n");
    return;
#endif
    anwcs_t* anwcs = NULL;
    tan_t* tan;
    char* tmpfile = create_temp_file("test-anwcs-wcs", NULL);
    double x, y, x2, y2, ra, dec, ra2, dec2;
    int ok, ok2;
    int i;

    // from http://live.astrometry.net/status.php?job=alpha-201004-29242410
    // test-anwcs-1.wcs
    const char* wcsliteral = 
        "SIMPLE  =                    T / Standard FITS file                             BITPIX  =                    8 / ASCII or bytes array                           NAXIS   =                    0 / Minimal header                                 EXTEND  =                    T / There may be FITS ext                          CTYPE1  = 'RA---TAN' / TAN (gnomic) projection                                  CTYPE2  = 'DEC--TAN' / TAN (gnomic) projection                                  WCSAXES =                    2 / no comment                                     EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         LONPOLE =                180.0 / no comment                                     LATPOLE =                  0.0 / no comment                                     CRVAL1  =        83.7131676182 / RA  of reference point                         CRVAL2  =       -5.10104333945 / DEC of reference point                         CRPIX1  =        221.593284607 / X reference pixel                              CRPIX2  =        169.655508041 / Y reference pixel                              CUNIT1  = 'deg     ' / X pixel scale units                                      CUNIT2  = 'deg     ' / Y pixel scale units                                      CD1_1   =    1.55258090814E-06 / Transformation matrix                          CD1_2   =     0.00081692280013 / no comment                                     CD2_1   =    -0.00081692280013 / no comment                                     CD2_2   =    1.55258090814E-06 / no comment                                     IMAGEW  =                  900 / Image width,  in pixels.                       IMAGEH  =                  600 / Image height, in pixels.                       DATE    = '2010-04-14T12:12:18' / Date this file was created.                   HISTORY Created by the Astrometry.net suite.                                    HISTORY For more details, see http://astrometry.net .                           HISTORY Subversion URL                                                          HISTORY   http://astrometry.net/svn/branches/astrometry/alpha/quads/            HISTORY Subversion revision 5409                                                HISTORY Subversion date 2007-10-09 13:49:13 -0400 (Tue, 09 Oct                  HISTORY   2007)                                                                 HISTORY This WCS header was created by Astrometry.net                           COMMENT -- solver parameters: --                                                COMMENT Index(0): /data1/INDEXES/200/index-219                                  COMMENT Index(1): /data1/INDEXES/200/index-218                                  COMMENT Index(2): /data1/INDEXES/200/index-217                                  COMMENT Index(3): /data1/INDEXES/200/index-216                                  COMMENT Index(4): /data1/INDEXES/200/index-215                                  COMMENT Index(5): /data1/INDEXES/200/index-214                                  COMMENT Index(6): /data1/INDEXES/200/index-213                                  COMMENT Index(7): /data1/INDEXES/200/index-212                                  COMMENT Index(8): /data1/INDEXES/200/index-211                                  COMMENT Index(9): /data1/INDEXES/200/index-210                                  COMMENT Index(10): /data1/INDEXES/200/index-209                                 COMMENT Index(11): /data1/INDEXES/200/index-208                                 COMMENT Index(12): /data1/INDEXES/200/index-207                                 COMMENT Index(13): /data1/INDEXES/200/index-206                                 COMMENT Index(14): /data1/INDEXES/200/index-205                                 COMMENT Index(15): /data1/INDEXES/200/index-204-00                              COMMENT Index(16): /data1/INDEXES/200/index-204-01                              COMMENT Index(17): /data1/INDEXES/200/index-204-02                              COMMENT Index(18): /data1/INDEXES/200/index-204-03                              COMMENT Index(19): /data1/INDEXES/200/index-204-04                              COMMENT Index(20): /data1/INDEXES/200/index-204-05                              COMMENT Index(21): /data1/INDEXES/200/index-204-06                              COMMENT Index(22): /data1/INDEXES/200/index-204-07                              COMMENT Index(23): /data1/INDEXES/200/index-204-08                              COMMENT Index(24): /data1/INDEXES/200/index-204-09                              COMMENT Index(25): /data1/INDEXES/200/index-204-10                              COMMENT Index(26): /data1/INDEXES/200/index-204-11                              COMMENT Index(27): /data1/INDEXES/200/index-203-00                              COMMENT Index(28): /data1/INDEXES/200/index-203-01                              COMMENT Index(29): /data1/INDEXES/200/index-203-02                              COMMENT Index(30): /data1/INDEXES/200/index-203-03                              COMMENT Index(31): /data1/INDEXES/200/index-203-04                              COMMENT Index(32): /data1/INDEXES/200/index-203-05                              COMMENT Index(33): /data1/INDEXES/200/index-203-06                              COMMENT Index(34): /data1/INDEXES/200/index-203-07                              COMMENT Index(35): /data1/INDEXES/200/index-203-08                              COMMENT Index(36): /data1/INDEXES/200/index-203-09                              COMMENT Index(37): /data1/INDEXES/200/index-203-10                              COMMENT Index(38): /data1/INDEXES/200/index-203-11                              COMMENT Index(39): /data1/INDEXES/200/index-202-00                              COMMENT Index(40): /data1/INDEXES/200/index-202-01                              COMMENT Index(41): /data1/INDEXES/200/index-202-02                              COMMENT Index(42): /data1/INDEXES/200/index-202-03                              COMMENT Index(43): /data1/INDEXES/200/index-202-04                              COMMENT Index(44): /data1/INDEXES/200/index-202-05                              COMMENT Index(45): /data1/INDEXES/200/index-202-06                              COMMENT Index(46): /data1/INDEXES/200/index-202-07                              COMMENT Index(47): /data1/INDEXES/200/index-202-08                              COMMENT Index(48): /data1/INDEXES/200/index-202-09                              COMMENT Index(49): /data1/INDEXES/200/index-202-10                              COMMENT Index(50): /data1/INDEXES/200/index-202-11                              COMMENT Index(51): /data1/INDEXES/200/index-201-00                              COMMENT Index(52): /data1/INDEXES/200/index-201-01                              COMMENT Index(53): /data1/INDEXES/200/index-201-02                              COMMENT Index(54): /data1/INDEXES/200/index-201-03                              COMMENT Index(55): /data1/INDEXES/200/index-201-04                              COMMENT Index(56): /data1/INDEXES/200/index-201-05                              COMMENT Index(57): /data1/INDEXES/200/index-201-06                              COMMENT Index(58): /data1/INDEXES/200/index-201-07                              COMMENT Index(59): /data1/INDEXES/200/index-201-08                              COMMENT Index(60): /data1/INDEXES/200/index-201-09                              COMMENT Index(61): /data1/INDEXES/200/index-201-10                              COMMENT Index(62): /data1/INDEXES/200/index-201-11                              COMMENT Index(63): /data1/INDEXES/200/index-200-00                              COMMENT Index(64): /data1/INDEXES/200/index-200-01                              COMMENT Index(65): /data1/INDEXES/200/index-200-02                              COMMENT Index(66): /data1/INDEXES/200/index-200-03                              COMMENT Index(67): /data1/INDEXES/200/index-200-04                              COMMENT Index(68): /data1/INDEXES/200/index-200-05                              COMMENT Index(69): /data1/INDEXES/200/index-200-06                              COMMENT Index(70): /data1/INDEXES/200/index-200-07                              COMMENT Index(71): /data1/INDEXES/200/index-200-08                              COMMENT Index(72): /data1/INDEXES/200/index-200-09                              COMMENT Index(73): /data1/INDEXES/200/index-200-10                              COMMENT Index(74): /data1/INDEXES/200/index-200-11                              COMMENT Field name: field.xy.fits                                               COMMENT Field scale lower: 0.4 arcsec/pixel                                     COMMENT Field scale upper: 720 arcsec/pixel                                     COMMENT X col name: X                                                           COMMENT Y col name: Y                                                           COMMENT Start obj: 0                                                            COMMENT End obj: 200                                                            COMMENT Solved_in: solved                                                       COMMENT Solved_out: solved                                                      COMMENT Solvedserver: (null)                                                    COMMENT Parity: 2                                                               COMMENT Codetol: 0.01                                                           COMMENT Verify distance: 0 arcsec                                               COMMENT Verify pixels: 1 pix                                                    COMMENT Maxquads: 0                                                             COMMENT Maxmatches: 0                                                           COMMENT Cpu limit: 0 s                                                          COMMENT Time limit: 0 s                                                         COMMENT Total time limit: 0 s                                                   COMMENT Total CPU limit: 600 s                                                  COMMENT Tweak: no                                                               COMMENT --                                                                      COMMENT -- properties of the matching quad: --                                  COMMENT quadno: 686636                                                          COMMENT stars: 1095617,1095660,1095623,1095618                                  COMMENT field: 6,5,24,35                                                        COMMENT code error: 0.00868071                                                  COMMENT noverlap: 42                                                            COMMENT nconflict: 1                                                            COMMENT nfield: 88                                                              COMMENT nindex: 139                                                             COMMENT scale: 2.94093 arcsec/pix                                               COMMENT parity: 1                                                               COMMENT quads tried: 2166080                                                    COMMENT quads matched: 2079562                                                  COMMENT quads verified: 1747182                                                 COMMENT objs tried: 0                                                           COMMENT cpu time: 117.82                                                        COMMENT --                                                                      AN_JOBID= 'alpha-201004-29242410' / Astrometry.net job ID                       END                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ";

    if (write_file(tmpfile, wcsliteral, strlen(wcsliteral))) {
        ERROR("Failed to write WCS to temp file %s", tmpfile);
        CuFail(tc, "failed to write WCS to temp file");
    }

    tan = tan_read_header_file(tmpfile, NULL);
    CuAssertPtrNotNull(tc, tan);

    for (i=0; i<2; i++) {
        if (i == 0) {
            anwcs = anwcs_open_wcslib(tmpfile, 0);
        } else if (i == 1) {
            anwcs = anwcs_open_sip(tmpfile, 0);
        }
        CuAssertPtrNotNull(tc, anwcs);

        /*
         printf("ANWCS:\n");
         anwcs_print(anwcs, stdout);
	
         printf("TAN:\n");
         tan_print_to(tan, stdout);
         */
        /* this wcs has:
         crval=(83.7132, -5.10104)
         crpix=(221.593, 169.656)
         CD = (   1.5526e-06     0.00081692 )
         (  -0.00081692     1.5526e-06 )
         */

        // check crval <-> crpix.
        x = tan->crpix[0];
        y = tan->crpix[1];
        ra = 1; ra2 = 2; dec = 3; dec2 = 4;

        tan_pixelxy2radec(tan, x, y, &ra, &dec);
        CuAssertDblEquals(tc, tan->crval[0], ra, 1e-6);
        CuAssertDblEquals(tc, tan->crval[1], dec, 1e-6);

        ok = anwcs_pixelxy2radec(anwcs, x, y, &ra2, &dec2);
        CuAssertIntEquals(tc, 0, ok);
        CuAssertDblEquals(tc, tan->crval[0], ra2, 1e-6);
        CuAssertDblEquals(tc, tan->crval[1], dec2, 1e-6);

        ra = tan->crval[0];
        dec = tan->crval[1];
        x = 1; x2 = 2; y = 3; y2 = 4;

        ok = tan_radec2pixelxy(tan, ra, dec, &x, &y);
        CuAssertIntEquals(tc, TRUE, ok);
        CuAssertDblEquals(tc, tan->crpix[0], x, 1e-6);
        CuAssertDblEquals(tc, tan->crpix[1], y, 1e-6);

        ok2 = anwcs_radec2pixelxy(anwcs, ra, dec, &x2, &y2);
        CuAssertIntEquals(tc, 0, ok2);
        CuAssertDblEquals(tc, tan->crpix[0], x2, 1e-6);
        CuAssertDblEquals(tc, tan->crpix[1], y2, 1e-6);

        // check pixel (0,0).
        x = y = 0.0;
        ra = 1; ra2 = 2; dec = 3; dec2 = 4;

        tan_pixelxy2radec(tan, x, y, &ra, &dec);
        ok = anwcs_pixelxy2radec(anwcs, x, y, &ra2, &dec2);
        CuAssertIntEquals(tc, 0, ok);

        CuAssertDblEquals(tc, ra, ra2, 1e-6);
        CuAssertDblEquals(tc, dec, dec2, 1e-6);

        // check RA,Dec (85, -4)
        ra = 85;
        dec = -4;
        x = 1; x2 = 2; y = 3; y2 = 4;

        ok = tan_radec2pixelxy(tan, ra, dec, &x, &y);
        CuAssertIntEquals(tc, TRUE, ok);
        ok2 = anwcs_radec2pixelxy(anwcs, ra, dec, &x2, &y2);
        CuAssertIntEquals(tc, 0, ok2);
        printf("x,y (%g,%g) vs (%g,%g)\n", x, y, x2, y2);
        CuAssertDblEquals(tc, x, x2, 1e-6);
        CuAssertDblEquals(tc, y, y2, 1e-6);

        anwcs_free(anwcs);
    }

    free(tan);
}


