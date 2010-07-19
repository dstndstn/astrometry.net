#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"
#include "tweak.h"
#include "sip.h"
#include "sip-utils.h"
#include "log.h"

void test_tweak_1(CuTest* tc) {
	int GX = 5;
	int GY = 5;
	double gridx[GX];
	double gridy[GY];
	double origxy[GX*GY*2];
	double xy[GX*GY*2];
	double radec[GX*GY*2];
	sip_t sip;
	tan_t* tan = &(sip.wcstan);
	int i,j;
	starxy_t* sxy;
	tweak_t* t;
	sip_t* outsip;

	log_init(LOG_VERB);

	t = tweak_new();
	memset(&sip, 0, sizeof(sip_t));

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

	sip.a_order = sip.b_order = 2;
	sip.a[2][0] = 10.*1e-6;
	sip.ap_order = sip.bp_order = 4;

	sip_compute_inverse_polynomials(&sip, 0, 0, 0, 0, 0, 0);

	for (i=0; i<GX; i++)
		gridx[i] = 0.5 + i * tan->imagew / (GX-1);
	for (i=0; i<GY; i++)
		gridy[i] = 0.5 + i * tan->imageh / (GY-1);

	printf("Input SIP:\n");
	sip_print(&sip);

	for (i=0; i<GY; i++)
		for (j=0; j<GX; j++) {
			double ra, dec, x, y;
			origxy[2*(i*GX+j) + 0] = gridx[j];
			origxy[2*(i*GX+j) + 1] = gridy[i];
			tan_pixelxy2radec(tan, gridx[j], gridy[i], &ra, &dec);
			radec[2*(i*GX+j) + 0] = ra;
			radec[2*(i*GX+j) + 1] = dec;
			sip_radec2pixelxy(&sip, ra, dec, &x, &y);
			xy[2*(i*GX+j) + 0] = x;
			xy[2*(i*GX+j) + 1] = y;
		}

	sxy = starxy_new(GX*GY, FALSE, FALSE);
	starxy_set_xy_array(sxy, xy);

	il* imcorr = il_new(256);
	il* refcorr = il_new(256);
	dl* weights = dl_new(256);
	for (i=0; i<(GX*GY); i++) {
		il_append(imcorr, i);
		il_append(refcorr, i);
		dl_append(weights, 1.0);
	}

	tweak_push_wcs_tan(t, tan);

	outsip = t->sip;
	outsip->a_order = outsip->b_order = sip.a_order;
	outsip->ap_order = outsip->bp_order = sip.ap_order;

	t->weighted_fit = TRUE;
	tweak_push_ref_ad_array(t, radec, GX*GY);
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
	sip_print(outsip);

	CuAssertDblEquals(tc, outsip->wcstan.crval[0], tan->crval[0], 1e-6);
	CuAssertDblEquals(tc, outsip->wcstan.crval[1], tan->crval[1], 1e-6);
	// should be exactly equal.
	CuAssertDblEquals(tc, outsip->wcstan.crpix[0], tan->crpix[0], 1e-10);
	CuAssertDblEquals(tc, outsip->wcstan.crpix[1], tan->crpix[1], 1e-10);

	CuAssertDblEquals(tc, outsip->wcstan.cd[0][0], tan->cd[0][0], 1e-10);
	CuAssertDblEquals(tc, outsip->wcstan.cd[0][1], tan->cd[0][1], 1e-10);
	CuAssertDblEquals(tc, outsip->wcstan.cd[1][0], tan->cd[1][0], 1e-10);
	CuAssertDblEquals(tc, outsip->wcstan.cd[1][1], tan->cd[1][1], 1e-10);

	CuAssertDblEquals(tc, outsip->wcstan.imagew, tan->imagew, 1e-10);
	CuAssertDblEquals(tc, outsip->wcstan.imageh, tan->imageh, 1e-10);

	double *d1, *d2;
	d1 = (double*)outsip->a;
	d2 = (double*)&(sip.a);
	for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
		CuAssertDblEquals(tc, d1[i], d2[i], 1e-10);
	d1 = (double*)outsip->b;
	d2 = (double*)&(sip.b);
	for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
		CuAssertDblEquals(tc, d1[i], d2[i], 1e-10);
	d1 = (double*)outsip->ap;
	d2 = (double*)&(sip.ap);
	for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
		CuAssertDblEquals(tc, d1[i], d2[i], 1e-10);
	d1 = (double*)outsip->bp;
	d2 = (double*)&(sip.bp);
	for (i=0; i<(SIP_MAXORDER * SIP_MAXORDER); i++)
		CuAssertDblEquals(tc, d1[i], d2[i], 1e-10);
	
	CuAssertIntEquals(tc, outsip->a_order, sip.a_order);
	CuAssertIntEquals(tc, outsip->b_order, sip.b_order);
	CuAssertIntEquals(tc, outsip->ap_order, sip.ap_order);
	CuAssertIntEquals(tc, outsip->bp_order, sip.bp_order);
}

void test_tchebytweak(CuTest* tc) {


}

