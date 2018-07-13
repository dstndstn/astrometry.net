/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

#include "cutest.h"
#include "tweak.h"
#include "sip.h"
#include "sip-utils.h"
#include "log.h"

#define GAUSSIAN_SAMPLE_INVALID -1e300

static double uniform_sample(double low, double high) {
    if (low == high) return low;
    return low + (high - low)*((double)rand() / (double)RAND_MAX);
}

static double gaussian_sample(double mean, double stddev) {
    // from http://www.taygeta.com/random/gaussian.html
    // Algorithm by Dr. Everett (Skip) Carter, Jr.
    static double y2 = GAUSSIAN_SAMPLE_INVALID;
    double x1, x2, w, y1;
    // this algorithm generates random samples in pairs; the INVALID
    // jibba-jabba stores the second value until the next time the
    // function is called.
    if (y2 != GAUSSIAN_SAMPLE_INVALID) {
        y1 = y2;
        y2 = GAUSSIAN_SAMPLE_INVALID;
        return mean + y1 * stddev;
    }
    do {
        x1 = uniform_sample(-1, 1);
        x2 = uniform_sample(-1, 1);
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );
    w = sqrt( (-2.0 * log(w)) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    return mean + y1 * stddev;
}

static void set_grid(int GX, int GY, tan_t* tan, sip_t* sip,
                     double* tanxy, double* radec, double* xy,
                     double* gridx, double* gridy) {
    int i, j;

    if (GX == 1)
        gridx[0] = 0.5 + 0.5 * tan->imagew;
    else
        for (i=0; i<GX; i++)
            gridx[i] = 0.5 + i * tan->imagew / (GX-1);
    if (GY == 1)
        gridy[0] = 0.5 + 0.5 * tan->imageh;
    else
        for (i=0; i<GY; i++)
            gridy[i] = 0.5 + i * tan->imageh / (GY-1);

    for (i=0; i<GY; i++)
        for (j=0; j<GX; j++) {
            double ra, dec, x, y;
            // Lay down the grid in (measured, distorted) CCD space
            x = xy[2*(i*GX+j) + 0] = gridx[j];
            y = xy[2*(i*GX+j) + 1] = gridy[i];
            sip_pixelxy2radec(sip, x, y, &ra, &dec);
            if(!tan_radec2pixelxy(tan, ra, dec, &x, &y)) perror("tan_radec2pixelxy==FALSE");
            tanxy[2*(i*GX+j) + 0] = x;
            tanxy[2*(i*GX+j) + 1] = y;
            radec[2*(i*GX+j) + 0] = ra;
            radec[2*(i*GX+j) + 1] = dec;
        }

}

static sip_t* run_test(CuTest* tc, sip_t* sip, int N, double* xy, double* radec) {
    int i;
    starxy_t* sxy;
    tweak_t* t;
    sip_t* outsip;
    il* imcorr;
    il* refcorr;
    dl* weights;
    tan_t* tan = &(sip->wcstan);

    printf("Input SIP:\n");
    sip_print_to(sip, stdout);
    fflush(NULL);

    sxy = starxy_new(N, FALSE, FALSE);
    starxy_set_xy_array(sxy, xy);

    imcorr = il_new(256);
    refcorr = il_new(256);
    weights = dl_new(256);
    for (i=0; i<N; i++) {
        il_append(imcorr, i);
        il_append(refcorr, i);
        dl_append(weights, 1.0);
    }

    t = tweak_new();
    tweak_push_wcs_tan(t, tan);

    outsip = t->sip;
    outsip->a_order = outsip->b_order = sip->a_order;
    outsip->ap_order = outsip->bp_order = sip->ap_order;

    t->weighted_fit = TRUE;
    tweak_push_ref_ad_array(t, radec, N);
    tweak_push_image_xy(t, sxy);
    tweak_push_correspondence_indices(t, imcorr, refcorr, NULL, weights);
    tweak_skip_shift(t);

    // push correspondences
    // push image xy
    // push ref ra,dec
    // push ref xy (tan)
    // push tan

    tweak_go_to(t, TWEAK_HAS_LINEAR_CD);

    printf("Output SIP:\n");
    sip_print_to(outsip, stdout);

    CuAssertDblEquals(tc, tan->imagew, outsip->wcstan.imagew, 1e-10);
    CuAssertDblEquals(tc, tan->imageh, outsip->wcstan.imageh, 1e-10);

    // should be exactly equal.
    CuAssertDblEquals(tc, tan->crpix[0], outsip->wcstan.crpix[0], 1e-10);
    CuAssertDblEquals(tc, tan->crpix[1], outsip->wcstan.crpix[1], 1e-10);

    t->sip = NULL;
    tweak_free(t);
    starxy_free(sxy);
    return outsip;
}
					 
void test_tweak_1(CuTest* tc) {
    int GX = 5;
    int GY = 5;
    double origxy[GX*GY*2];
    double xy[GX*GY*2];
    double radec[GX*GY*2];
    double gridx[GX];
    double gridy[GY];
    sip_t thesip;
    sip_t* sip = &thesip;
    tan_t* tan = &(sip->wcstan);
    int i;
    sip_t* outsip;

    printf("\ntest_tweak_1\n\n");

    log_init(LOG_VERB);

    memset(sip, 0, sizeof(sip_t));

    tan->imagew = 2000;
    tan->imageh = 2000;
    tan->crval[0] = 150;
    tan->crval[1] = -30;
    tan->crpix[0] = 1000.5;
    tan->crpix[1] = 1000.5;
    tan->cd[0][0] = 1./1000.;
    tan->cd[0][1] = 0;
    tan->cd[1][1] = 1./1000.;
    tan->cd[1][0] = 0;

    sip->a_order = sip->b_order = 2;
    sip->a[2][0] = 10.*1e-6;
    sip->ap_order = sip->bp_order = 4;

    sip_compute_inverse_polynomials(sip, 0, 0, 0, 0, 0, 0);

    /*
     printf("After compute_inverse_polynomials:\n");
     sip_print_to(sip, stdout);
     fflush(NULL);
     */

    set_grid(GX, GY, tan, sip, origxy, radec, xy, gridx, gridy);

    /*{
     int i,j;
     printf("RA,Dec\n");
     for (i=0; i<GY; i++) {
     for (j=0; j<GX; j++) {
     fflush(NULL);
     printf("gy %i gyx %i: %g %g\n", i, j, radec[2*(i*GX+j)],
     radec[2*(i*GX+j) + 1]);
     fflush(NULL);
     }
     }
     }*/

    outsip = run_test(tc, sip, GX*GY, xy, radec);

    CuAssertDblEquals(tc, tan->crval[0], outsip->wcstan.crval[0], 1e-6);
    CuAssertDblEquals(tc, tan->crval[1], outsip->wcstan.crval[1], 1e-6);

    CuAssertDblEquals(tc, tan->cd[0][0], outsip->wcstan.cd[0][0], 1e-10);
    CuAssertDblEquals(tc, tan->cd[0][1], outsip->wcstan.cd[0][1], 1e-14);
    CuAssertDblEquals(tc, tan->cd[1][0], outsip->wcstan.cd[1][0], 1e-14);
    CuAssertDblEquals(tc, tan->cd[1][1], outsip->wcstan.cd[1][1], 1e-10);

    double *d1, *d2;
    d1 = (double*)outsip->a;
    d2 = (double*)&(sip->a);
    for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
        CuAssertDblEquals(tc, d2[i], d1[i], 1e-13);
    d1 = (double*)outsip->b;
    d2 = (double*)&(sip->b);
    for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++) {
        printf("test_tweak_1: Expecting %.18g, got %.18g\n", d2[i], d1[i]);
        fflush(NULL);
        CuAssertDblEquals(tc, d2[i], d1[i], 3e-18);
    }
    d1 = (double*)outsip->ap;
    d2 = (double*)&(sip->ap);
    for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
        CuAssertDblEquals(tc, d2[i], d1[i], 1e-10);
    d1 = (double*)outsip->bp;
    d2 = (double*)&(sip->bp);
    for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
        CuAssertDblEquals(tc, d2[i], d1[i], 1e-10);
    CuAssertIntEquals(tc, sip->a_order, outsip->a_order);
    CuAssertIntEquals(tc, sip->b_order, outsip->b_order);
    CuAssertIntEquals(tc, sip->ap_order, outsip->ap_order);
    CuAssertIntEquals(tc, sip->bp_order, outsip->bp_order);
}


static void tst_tweak_n(CuTest* tc, int run, int GX, int GY) {
    double origxy[GX*GY*2];
    double xy[GX*GY*2];
    double noisyxy[GX*GY*2];
    double radec[GX*GY*2];
    double gridx[GX];
    double gridy[GY];
    sip_t thesip;
    sip_t* sip = &thesip;
    tan_t* tan = &(sip->wcstan);
    int i,j;
    sip_t* outsip;

    printf("\ntest_tweak_%i\n\n", run);

    log_init(LOG_VERB);

    memset(sip, 0, sizeof(sip_t));

    tan->imagew = 2000;
    tan->imageh = 2000;
    tan->crval[0] = 150;
    tan->crval[1] = -30;
    tan->crpix[0] = 1000.5;
    tan->crpix[1] = 1000.5;
    tan->cd[0][0] = 1./1000.;
    tan->cd[0][1] = 0;
    tan->cd[1][1] = 1./1000.;
    tan->cd[1][0] = 0;

    sip->a_order = sip->b_order = 2;
    sip->a[2][0] = 10.*1e-6;
    sip->b[0][2] = -10.*1e-6;
    sip->ap_order = sip->bp_order = 4;

    sip_compute_inverse_polynomials(sip, 0, 0, 0, 0, 0, 0);

    set_grid(GX, GY, tan, sip, origxy, radec, xy, gridx, gridy);

    // add noise to observed xy positions.
    srand(42);
    for (i=0; i<(GX*GY*2); i++) {
        noisyxy[i] = xy[i] + gaussian_sample(0.0, 1.0);
    }

    fprintf(stderr, "from numpy import array\n");
    fprintf(stderr, "x0,y0 = %g,%g\n", tan->crpix[0], tan->crpix[1]);
    fprintf(stderr, "gridx_%i=array([", run);
    for (i=0; i<GX; i++)
        fprintf(stderr, "%g, ", gridx[i]);
    fprintf(stderr, "])\n");
    fprintf(stderr, "gridy_%i=array([", run);
    for (i=0; i<GY; i++)
        fprintf(stderr, "%g, ", gridy[i]);
    fprintf(stderr, "])\n");
    fprintf(stderr, "origxy_%i = array([", run);
    for (i=0; i<(GX*GY); i++)
        fprintf(stderr, "[%g,%g],", origxy[2*i+0], origxy[2*i+1]);
    fprintf(stderr, "])\n");
    fprintf(stderr, "xy_%i = array([", run);
    for (i=0; i<(GX*GY); i++)
        fprintf(stderr, "[%g,%g],", xy[2*i+0], xy[2*i+1]);
    fprintf(stderr, "])\n");

    fprintf(stderr, "noisyxy_%i = array([", run);
    for (i=0; i<(GX*GY); i++)
        fprintf(stderr, "[%g,%g],", noisyxy[2*i+0], noisyxy[2*i+1]);
    fprintf(stderr, "])\n");

    fprintf(stderr, "truesip_a_%i = array([", run);
    for (i=0; i<=sip->a_order; i++)
        for (j=0; j<=sip->a_order; j++)
            if (sip->a[i][j] != 0)
                fprintf(stderr, "[%i,%i,%g],", i, j, sip->a[i][j]);
    fprintf(stderr, "])\n");

    fprintf(stderr, "truesip_b_%i = array([", run);
    for (i=0; i<=sip->a_order; i++)
        for (j=0; j<=sip->a_order; j++)
            if (sip->b[i][j] != 0)
                fprintf(stderr, "[%i,%i,%g],", i, j, sip->b[i][j]);
    fprintf(stderr, "])\n");

    fprintf(stderr, "sip_a_%i = {}\n", run);
    fprintf(stderr, "sip_b_%i = {}\n", run);

    int o;
    for (o=2; o<6; o++) {
        sip->a_order = o;
        outsip = run_test(tc, sip, GX*GY, noisyxy, radec);

        fprintf(stderr, "sip_a_%i[%i] = array([", run, o);
        for (i=0; i<=outsip->a_order; i++)
            for (j=0; j<=outsip->a_order; j++)
                if (outsip->a[i][j] != 0)
                    fprintf(stderr, "[%i,%i,%g],", i, j, outsip->a[i][j]);
        fprintf(stderr, "])\n");

        fprintf(stderr, "sip_b_%i[%i] = array([", run, o);
        for (i=0; i<=outsip->a_order; i++)
            for (j=0; j<=outsip->a_order; j++)
                if (outsip->b[i][j] != 0)
                    fprintf(stderr, "[%i,%i,%g],", i, j, outsip->b[i][j]);
        fprintf(stderr, "])\n");
    }

    sip->a_order = 2;
    outsip = run_test(tc, sip, GX*GY, noisyxy, radec);

    CuAssertDblEquals(tc, tan->crval[0], outsip->wcstan.crval[0], 1e-3);
    CuAssertDblEquals(tc, tan->crval[1], outsip->wcstan.crval[1], 1e-3);

    CuAssertDblEquals(tc, tan->cd[0][0], outsip->wcstan.cd[0][0], 1e-6);
    CuAssertDblEquals(tc, tan->cd[0][1], outsip->wcstan.cd[0][1], 1e-6);
    CuAssertDblEquals(tc, tan->cd[1][0], outsip->wcstan.cd[1][0], 1e-6);
    CuAssertDblEquals(tc, tan->cd[1][1], outsip->wcstan.cd[1][1], 1e-6);

    if (run == 2) {
        double *d1, *d2;
        d1 = (double*)outsip->a;
        d2 = (double*)&(sip->a);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
            // rather large error, no?
            CuAssertDblEquals(tc, d2[i], d1[i], 6e-7);
        d1 = (double*)outsip->b;
        d2 = (double*)&(sip->b);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++) {
            printf("test_tweak_2, run 2: Expecting %.18g, got %.18g\n", d2[i], d1[i]);
            fflush(NULL);
            CuAssertDblEquals(tc, d2[i], d1[i], 3e-7);
        }
        d1 = (double*)outsip->ap;
        d2 = (double*)&(sip->ap);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
            CuAssertDblEquals(tc, d2[i], d1[i], 1e-6);
        d1 = (double*)outsip->bp;
        d2 = (double*)&(sip->bp);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
            CuAssertDblEquals(tc, d2[i], d1[i], 1e-6);
        CuAssertIntEquals(tc, sip->a_order, outsip->a_order);
        CuAssertIntEquals(tc, sip->b_order, outsip->b_order);
        CuAssertIntEquals(tc, sip->ap_order, outsip->ap_order);
        CuAssertIntEquals(tc, sip->bp_order, outsip->bp_order);
    } else if (run == 3) {
        double *d1, *d2;
        d1 = (double*)outsip->a;
        d2 = (double*)&(sip->a);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++) {
            // rather large error, no?
            printf("test_tweak_2, run 3: Expecting %.18g, got %.18g\n", d2[i], d1[i]);
            fflush(NULL);
            CuAssertDblEquals(tc, d2[i], d1[i], 7e-7);
        }
        d1 = (double*)outsip->b;
        d2 = (double*)&(sip->b);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++) {
            printf("test_tweak_2, run 3b: Expecting %.18g, got %.18g\n", d2[i], d1[i]);
            fflush(NULL);
            CuAssertDblEquals(tc, d2[i], d1[i], 2e-6);
        }
        d1 = (double*)outsip->ap;
        d2 = (double*)&(sip->ap);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
            CuAssertDblEquals(tc, d2[i], d1[i], 1e-6);
        d1 = (double*)outsip->bp;
        d2 = (double*)&(sip->bp);
        for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++) {
            printf("test_tweak_2, run 3c: Expecting %.18g, got %.18g\n", d2[i], d1[i]);
            fflush(NULL);
            CuAssertDblEquals(tc, d2[i], d1[i], 2e-6);
        }
        CuAssertIntEquals(tc, sip->a_order, outsip->a_order);
        CuAssertIntEquals(tc, sip->b_order, outsip->b_order);
        CuAssertIntEquals(tc, sip->ap_order, outsip->ap_order);
        CuAssertIntEquals(tc, sip->bp_order, outsip->bp_order);
    }
}


/*
 void tst_tweak_3(CuTest* tc) {
 int GX = 11;
 int GY = 1;
 double origxy[GX*GY*2];
 double xy[GX*GY*2];
 double noisyxy[GX*GY*2];
 double radec[GX*GY*2];
 double gridx[GX];
 double gridy[GY];
 sip_t thesip;
 sip_t* sip = &thesip;
 tan_t* tan = &(sip->wcstan);
 int i,j;
 sip_t* outsip;

 printf("\ntest_tweak_3\n\n");

 log_init(LOG_VERB);

 memset(sip, 0, sizeof(sip_t));

 tan->imagew = 2000;
 tan->imageh = 2000;
 tan->crval[0] = 150;
 tan->crval[1] = -30;
 tan->crpix[0] = 1000.5;
 tan->crpix[1] = 1000.5;
 tan->cd[0][0] = 1./1000.;
 tan->cd[0][1] = 0;
 tan->cd[1][1] = 1./1000.;
 tan->cd[1][0] = 0;

 sip->a_order = sip->b_order = 2;
 sip->a[2][0] = 10.*1e-6;
 sip->ap_order = sip->bp_order = 4;

 sip_compute_inverse_polynomials(sip, 0, 0, 0, 0, 0, 0);

 set_grid(GX, GY, tan, sip, origxy, radec, xy, gridx, gridy);

 // add noise to observed y positions only.
 srand(42);
 for (i=0; i<(GX*GY); i++) {
 noisyxy[2*i+0] = xy[2*i+0];
 noisyxy[2*i+1] = xy[2*i+1] + gaussian_sample(0.0, 1.0);
 }

 outsip = run_test(tc, sip, GX*GY, noisyxy, radec);

 fprintf(stderr, "gridx_3=array([");
 for (i=0; i<GX; i++)
 fprintf(stderr, "%g, ", gridx[i]);
 fprintf(stderr, "])\n");
 fprintf(stderr, "gridy_3=array([");
 for (i=0; i<GY; i++)
 fprintf(stderr, "%g, ", gridy[i]);
 fprintf(stderr, "])\n");
 fprintf(stderr, "origxy_3 = array([");
 for (i=0; i<(GX*GY); i++)
 fprintf(stderr, "[%g,%g],", origxy[2*i+0], origxy[2*i+1]);
 fprintf(stderr, "])\n");
 fprintf(stderr, "xy_3 = array([");
 for (i=0; i<(GX*GY); i++)
 fprintf(stderr, "[%g,%g],", xy[2*i+0], xy[2*i+1]);
 fprintf(stderr, "])\n");

 fprintf(stderr, "noisyxy_3 = array([");
 for (i=0; i<(GX*GY); i++)
 fprintf(stderr, "[%g,%g],", noisyxy[2*i+0], noisyxy[2*i+1]);
 fprintf(stderr, "])\n");

 fprintf(stderr, "truesip_a_3 = array([");
 for (i=0; i<=sip->a_order; i++)
 for (j=0; j<=sip->a_order; j++)
 if (sip->a[i][j] != 0)
 fprintf(stderr, "[%i,%i,%g],", i, j, sip->a[i][j]);
 fprintf(stderr, "])\n");

 fprintf(stderr, "truesip_b_3 = array([");
 for (i=0; i<=sip->a_order; i++)
 for (j=0; j<=sip->a_order; j++)
 if (sip->b[i][j] != 0)
 fprintf(stderr, "[%i,%i,%g],", i, j, sip->b[i][j]);
 fprintf(stderr, "])\n");

 fprintf(stderr, "sip_a_3 = array([");
 for (i=0; i<=outsip->a_order; i++)
 for (j=0; j<=outsip->a_order; j++)
 if (outsip->a[i][j] != 0)
 fprintf(stderr, "[%i,%i,%g],", i, j, outsip->a[i][j]);
 fprintf(stderr, "])\n");

 fprintf(stderr, "sip_b_3 = array([");
 for (i=0; i<=outsip->a_order; i++)
 for (j=0; j<=outsip->a_order; j++)
 if (outsip->b[i][j] != 0)
 fprintf(stderr, "[%i,%i,%g],", i, j, outsip->b[i][j]);
 fprintf(stderr, "])\n");

 CuAssertDblEquals(tc, tan->crval[0], outsip->wcstan.crval[0], 1e-3);
 CuAssertDblEquals(tc, tan->crval[1], outsip->wcstan.crval[1], 1e-3);

 CuAssertDblEquals(tc, tan->cd[0][0], outsip->wcstan.cd[0][0], 1e-6);
 CuAssertDblEquals(tc, tan->cd[0][1], outsip->wcstan.cd[0][1], 1e-6);
 CuAssertDblEquals(tc, tan->cd[1][0], outsip->wcstan.cd[1][0], 1e-6);
 CuAssertDblEquals(tc, tan->cd[1][1], outsip->wcstan.cd[1][1], 1e-6);

 double *d1, *d2;
 d1 = (double*)outsip->a;
 d2 = (double*)&(sip->a);
 for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
 // rather large error, no?
 CuAssertDblEquals(tc, d2[i], d1[i], 6e-7);
 d1 = (double*)outsip->b;
 d2 = (double*)&(sip->b);
 for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
 CuAssertDblEquals(tc, d2[i], d1[i], 2e-7);
 d1 = (double*)outsip->ap;
 d2 = (double*)&(sip->ap);
 for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
 CuAssertDblEquals(tc, d2[i], d1[i], 1e-6);
 d1 = (double*)outsip->bp;
 d2 = (double*)&(sip->bp);
 for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
 CuAssertDblEquals(tc, d2[i], d1[i], 1e-6);
 CuAssertIntEquals(tc, sip->a_order, outsip->a_order);
 CuAssertIntEquals(tc, sip->b_order, outsip->b_order);
 CuAssertIntEquals(tc, sip->ap_order, outsip->ap_order);
 CuAssertIntEquals(tc, sip->bp_order, outsip->bp_order);

 }
 */


void test_tweak_2(CuTest* tc) {
    tst_tweak_n(tc, 2, 11, 11);
    tst_tweak_n(tc, 3, 5, 5);
}




void test_tchebytweak(CuTest* tc) {


}

