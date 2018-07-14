/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "fit-wcs.h"
#include "sip.h"

//sip_t* wcs_shift(sip_t* wcs, double xs, double ys);

void test_wcs_shift(CuTest* tc) {
    sip_t wcs;
    memset(&wcs, 0, sizeof(sip_t));
    wcs.wcstan.crpix[0] = 324.867;
    wcs.wcstan.crpix[1] = 476.596;
    wcs.wcstan.crval[0] = 39.0268;
    wcs.wcstan.crval[1] = 65.0062;
    wcs.wcstan.cd[0][0] =  0.00061453;
    wcs.wcstan.cd[0][1] = -0.0035865;
    wcs.wcstan.cd[1][0] = -0.0035971;
    wcs.wcstan.cd[1][1] = -0.00061653;
    wcs.a[0][2] = 3.7161e-06;
    wcs.a[1][1] = 2.4926e-06;
    wcs.a[2][0] = -1.9189e-05;
    wcs.b[0][2] = -3.0798e-05;
    wcs.b[1][1] = 1.8929e-07;
    wcs.b[2][0] = 8.5835e-06;

    wcs.ap[0][1] = -3.7374e-05;
    wcs.ap[0][2] = -3.8391e-06;
    wcs.ap[1][0] = 3.582e-05;
    wcs.ap[1][1] = -2.5333e-06;
    wcs.ap[2][0] = 1.9578e-05;
    wcs.bp[0][1] = 0.00028454;
    wcs.bp[0][2] = 3.0996e-05;
    wcs.bp[1][0] = -1.0094e-05;
    wcs.bp[1][1] = -3.7012e-07;
    wcs.bp[2][0] = -8.7938e-06;
    wcs.wcstan.imagew = 1024;
    wcs.wcstan.imageh = 1024;
    wcs.a_order = wcs.b_order = 2;
    wcs.ap_order = wcs.bp_order = 2;

    sip_t wcs2;
    memset(&wcs2, 0, sizeof(sip_t));
    memcpy(&(wcs2.wcstan), &(wcs.wcstan), sizeof(tan_t));

#if 0
    //sip_t* newwcs = wcs_shift(&wcs, 2.52369e-05,1.38956e-05);
    sip_t* newwcs = wcs_shift(&wcs, 10., 10.);
    printf("New1:\n");
    sip_print_to(newwcs, stdout);
    printf("\n");
#else
    wcs_shift(&(wcs.wcstan), 10., 10.);
    printf("New1:\n");
    tan_print_to(&(wcs.wcstan), stdout);
    printf("\n");
#endif

#if 0
    sip_t* newwcs2 = wcs_shift(&wcs2, 10., 10.);
    printf("New2:\n");
    sip_print_to(newwcs2, stdout);
    printf("\n");
#endif
    // 10,10:
    
#if 0
    tan_t* newtan = &(newwcs->wcstan);
#else
    tan_t* newtan = &(wcs.wcstan);
#endif
    CuAssertDblEquals(tc, 39.0973, newtan->crval[0], 1e-4);
    CuAssertDblEquals(tc, 65.0483, newtan->crval[1], 1e-4);
    CuAssertDblEquals(tc, 324.867, newtan->crpix[0], 1e-3);
    CuAssertDblEquals(tc, 476.596, newtan->crpix[1], 1e-3);
    CuAssertDblEquals(tc, 0.00061053, newtan->cd[0][0], 1e-8);
    CuAssertDblEquals(tc, -0.0035872, newtan->cd[0][1], 1e-7);
    CuAssertDblEquals(tc, -0.0035978, newtan->cd[1][0], 1e-7);
    CuAssertDblEquals(tc, -0.00061252, newtan->cd[1][1], 1e-8);

    /*
     1,1 shift:
     TAN-SIP Structure:
     crval=(39.0338, 65.0104)
     crpix=(324.867, 476.596)
     CD = (   0.00061413     -0.0035866 )
     (   -0.0035972    -0.00061613 )
     image size = (1024 x 1024)
     SIP order: A=2, B=2, AP=2, BP=2
     A =            0           0  3.7161e-06
     0  2.4926e-06
     -1.9189e-05
     B =            0           0 -3.0798e-05
     0  1.8929e-07
     8.5835e-06
     AP =            0 -3.7374e-05 -3.8391e-06
     3.582e-05 -2.5333e-06
     1.9578e-05
     BP =            0  0.00028454  3.0996e-05
     -1.0094e-05 -3.7012e-07
     -8.7938e-06
     sqrt(det(CD))=13.119 [arcsec]
     .

     OK (1 test)



     TAN-SIP Structure:
     crval=(39.0268, 65.0062)
     crpix=(324.867, 476.596)
     CD = (   0.00061453     -0.0035865 )
     (   -0.0035971    -0.00061653 )
     image size = (1024 x 1024)
     SIP order: A=2, B=2, AP=2, BP=2
     A =            0           0  3.7161e-06
     0  2.4926e-06
     -1.9189e-05
     B =            0           0 -3.0798e-05
     0  1.8929e-07
     8.5835e-06
     AP =            0 -3.7374e-05 -3.8391e-06
     3.582e-05 -2.5333e-06
     1.9578e-05
     BP =            0  0.00028454  3.0996e-05
     -1.0094e-05 -3.7012e-07
     -8.7938e-06
     sqrt(det(CD))=13.119 [arcsec]
     .



     crval=(39.0268, 65.0062)
     crpix=(324.867, 476.596)
     CD = (   0.00061453     -0.0035865 )
     (   -0.0035971    -0.00061653 )
     image size = (1024 x 1024)
     SIP order: A=2, B=2, AP=2, BP=2
     A =            0           0  3.7161e-06
     0  2.4926e-06
     -1.9189e-05
     B =            0           0 -3.0798e-05
     0  1.8929e-07
     8.5835e-06
     AP =            0 -3.7374e-05 -3.8391e-06
     3.582e-05 -2.5333e-06
     1.9578e-05
     BP =            0  0.00028454  3.0996e-05
     -1.0094e-05 -3.7012e-07
     -8.7938e-06
     */
}

#if 0
int main() {
    CuString *output = CuStringNew();
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, test_wcs_shift);
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s\n", output->buffer);
    CuSuiteFree(suite);
    CuStringFree(output);
}
#endif
