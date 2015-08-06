/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"

static const char* wcsfile = "SIMPLE  =                    T / Standard FITS file                             BITPIX  =                    8 / ASCII or bytes array                           NAXIS   =                    0 / Minimal header                                 EXTEND  =                    T / There may be FITS ext                          CTYPE1  = 'RA---TAN-SIP' / TAN (gnomic) projection + SIP distortions            CTYPE2  = 'DEC--TAN-SIP' / TAN (gnomic) projection + SIP distortions            WCSAXES =                    2 / no comment                                     EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         LONPOLE =                180.0 / no comment                                     LATPOLE =                  0.0 / no comment                                     CRVAL1  =        11.5705189886 / RA  of reference point                         CRVAL2  =        42.1541506988 / DEC of reference point                         CRPIX1  =                 2048 / X reference pixel                              CRPIX2  =                 1024 / Y reference pixel                              CUNIT1  = 'deg     ' / X pixel scale units                                      CUNIT2  = 'deg     ' / Y pixel scale units                                      CD1_1   =    7.78009863032E-06 / Transformation matrix                          CD1_2   =    -1.0992330198E-05 / no comment                                     CD2_1   =   -1.14560595236E-05 / no comment                                     CD2_2   =   -8.63206896621E-06 / no comment                                     IMAGEW  =                 4096 / Image width,  in pixels.                       IMAGEH  =                 2048 / Image height, in pixels.                       A_ORDER =                    4 / Polynomial order, axis 1                       A_0_2   =    2.16626045427E-06 / no comment                                     A_0_3   =    8.43135826028E-12 / no comment                                     A_0_4   =    1.27723787676E-14 / no comment                                     A_1_1   =   -5.20376831571E-06 / no comment                                     A_1_2   =    -5.2962390408E-10 / no comment                                     A_1_3   =   -1.75526102672E-14 / no comment                                     A_2_0   =     8.5443232652E-06 / no comment                                     A_2_1   =   -4.30755974621E-11 / no comment                                     A_2_2   =    3.82502701466E-14 / no comment                                     A_3_0   =    -4.7567645697E-10 / no comment                                     A_3_1   =    6.11248660507E-15 / no comment                                     A_4_0   =    2.60134165707E-14 / no comment                                     B_ORDER =                    4 / Polynomial order, axis 2                       B_0_2   =   -7.23056869993E-06 / no comment                                     B_0_3   =   -4.21356193854E-10 / no comment                                     B_0_4   =    2.93970053558E-15 / no comment                                     B_1_1   =    6.17195785471E-06 / no comment                                     B_1_2   =   -6.69823252817E-11 / no comment                                     B_1_3   =    1.83536133989E-14 / no comment                                     B_2_0   =   -1.74786318896E-06 / no comment                                     B_2_1   =   -5.15555867797E-10 / no comment                                     B_2_2   =   -2.78970082125E-14 / no comment                                     B_3_0   =    8.45057919961E-11 / no comment                                     B_3_1   =    2.40980945623E-16 / no comment                                     B_4_0   =   -1.72877462519E-14 / no comment                                     AP_ORDER=                    0 / Inv polynomial order, axis 1                   BP_ORDER=                    0 / Inv polynomial order, axis 2                   END                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ";

void test_compute_inverse(CuTest* tc) {
	double x,y, dx, dy;
	sip_t* wcs = sip_from_string(wcsfile, 0, NULL);
	CuAssertPtrNotNull(tc, wcs);
	printf("Read:\n");
	sip_print_to(wcs, stdout);

	CuAssertIntEquals(tc, 4, wcs->a_order);
	CuAssertIntEquals(tc, 4, wcs->b_order);
	CuAssertIntEquals(tc, 0, wcs->ap_order);
	CuAssertIntEquals(tc, 0, wcs->bp_order);
	CuAssertIntEquals(tc, 0, sip_ensure_inverse_polynomials(wcs));

	printf("After ensuring inverse:\n");
	sip_print_to(wcs, stdout);

	CuAssertIntEquals(tc, 4, wcs->a_order);
	CuAssertIntEquals(tc, 4, wcs->b_order);
	CuAssertIntEquals(tc, 5, wcs->ap_order);
	CuAssertIntEquals(tc, 5, wcs->bp_order);

	dx = dy = 100;
	for (y=0; y<=sip_imageh(wcs); y+=dy) {
		for (x=0; x<=sip_imagew(wcs); x+=dx) {
			double ra,dec;
			double x2, y2;
            anbool ok;
			sip_pixelxy2radec(wcs, x, y, &ra, &dec);
			ok = sip_radec2pixelxy(wcs, ra, dec, &x2, &y2);
            CuAssertTrue(tc, ok);
			CuAssertDblEquals(tc, x, x2, 1e-2);
			CuAssertDblEquals(tc, y, y2, 1e-2);
			printf("x,y %g,%g --> error %g,%g\n", x,y, x2-x, y2-y);
		}
	}

	sip_free(wcs);
}


