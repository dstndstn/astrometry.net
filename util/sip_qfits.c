/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Keir Mierle, David W. Hogg, Sam Roweis and Dustin Lang.

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

#include <math.h>
#include <errno.h>
#include <string.h>

#include "sip_qfits.h"
#include "an-bool.h"
#include "fitsioutils.h"

int tan_write_to_file(const tan_t* tan, const char* fn) {
	FILE* fid;
	qfits_header* hdr;
	int res;
	fid = fopen(fn, "wb");
	if (!fid) {
		fprintf(stderr, "Failed to open file %s to write WCS header: %s\n", fn, strerror(errno));
		return -1;
	}
	hdr = tan_create_header(tan);
	if (!hdr) {
		fprintf(stderr, "Failed to create FITS header from WCS.\n");
		return -1;
	}
	res = qfits_header_dump(hdr, fid);
	qfits_header_destroy(hdr);
	if (res) {
		fprintf(stderr, "Failed to write FITS header to file %s: %s\n", fn, strerror(errno));
		return -1;
	}
	if (fclose(fid)) {
		fprintf(stderr, "Failed to close file %s after writing WCS header: %s\n", fn, strerror(errno));
		return -1;
	}
	return 0;
}

static void wcs_hdr_common(qfits_header* hdr, const tan_t* tan) {
	qfits_header_add(hdr, "WCSAXES", "2", NULL, NULL);
	qfits_header_add(hdr, "EQUINOX", "2000.0", "Equatorial coordinates definition (yr)", NULL);
	qfits_header_add(hdr, "LONPOLE", "180.0", NULL, NULL);
	qfits_header_add(hdr, "LATPOLE", "0.0", NULL, NULL);

	fits_header_add_double(hdr, "CRVAL1", tan->crval[0], "RA  of reference point");
	fits_header_add_double(hdr, "CRVAL2", tan->crval[1], "DEC of reference point");
	fits_header_add_double(hdr, "CRPIX1", tan->crpix[0], "X reference pixel");
	fits_header_add_double(hdr, "CRPIX2", tan->crpix[1], "Y reference pixel");
	qfits_header_add(hdr, "CUNIT1", "deg", "X pixel scale units", NULL);
	qfits_header_add(hdr, "CUNIT2", "deg", "Y pixel scale units", NULL);

	fits_header_add_double(hdr, "CD1_1", tan->cd[0][0], "Transformation matrix");
	fits_header_add_double(hdr, "CD1_2", tan->cd[0][1], "");
	fits_header_add_double(hdr, "CD2_1", tan->cd[1][0], "");
	fits_header_add_double(hdr, "CD2_2", tan->cd[1][1], "");

    if (tan->imagew > 0.0)
        fits_header_add_double(hdr, "IMAGEW", tan->imagew, "Image width,  in pixels.");
    if (tan->imageh > 0.0)
        fits_header_add_double(hdr, "IMAGEH", tan->imageh, "Image height, in pixels.");
}

static void add_polynomial(qfits_header* hdr, const char* format,
						   int order, const double* data, int datastride,
						   bool drop_linear) {
	int i, j;
	char key[64];
	for (i=0; i<=order; i++)
		for (j=0; (i+j)<=order; j++) {
			if (i+j < 1)
				continue;
			if (drop_linear && (i+j < 2))
				continue;
			sprintf(key, format, i, j);
			fits_header_add_double(hdr, key, data[i*datastride + j], "");
		}
}

void sip_add_to_header(qfits_header* hdr, const sip_t* sip) {
	qfits_header_add(hdr, "CTYPE1", "RA---TAN-SIP", "TAN (gnomic) projection + SIP distortions", NULL);
	qfits_header_add(hdr, "CTYPE2", "DEC--TAN-SIP", "TAN (gnomic) projection + SIP distortions", NULL);

	wcs_hdr_common(hdr, &(sip->wcstan));

	fits_header_add_int(hdr, "A_ORDER", sip->a_order, "Polynomial order, axis 1");
	add_polynomial(hdr, "A_%i_%i", sip->a_order, (double*)sip->a, SIP_MAXORDER, TRUE);

	fits_header_add_int(hdr, "B_ORDER", sip->b_order, "Polynomial order, axis 2");
	add_polynomial(hdr, "B_%i_%i", sip->b_order, (double*)sip->b, SIP_MAXORDER, TRUE);

	fits_header_add_int(hdr, "AP_ORDER", sip->ap_order, "Inv polynomial order, axis 1");
	add_polynomial(hdr, "AP_%i_%i", sip->ap_order, (double*)sip->ap, SIP_MAXORDER, FALSE);

	fits_header_add_int(hdr, "BP_ORDER", sip->bp_order, "Inv polynomial order, axis 2");
	add_polynomial(hdr, "BP_%i_%i", sip->bp_order, (double*)sip->bp, SIP_MAXORDER, FALSE);
}

qfits_header* sip_create_header(const sip_t* sip) {
	qfits_header* hdr = qfits_table_prim_header_default();
	sip_add_to_header(hdr, sip);
	return hdr;
}

void tan_add_to_header(qfits_header* hdr, const tan_t* tan) {
	qfits_header_add(hdr, "CTYPE1", "RA---TAN", "TAN (gnomic) projection", NULL);
	qfits_header_add(hdr, "CTYPE2", "DEC--TAN", "TAN (gnomic) projection", NULL);
	wcs_hdr_common(hdr, tan);
}

qfits_header* tan_create_header(const tan_t* tan) {
	qfits_header* hdr = qfits_table_prim_header_default();
	tan_add_to_header(hdr, tan);
	return hdr;
}

static void* read_header_file(const char* fn, void* dest,
							  void* (*readfunc)(const qfits_header*, void*)) {
	qfits_header* hdr;
	void* result;
	hdr = qfits_header_read(fn);
	if (!hdr) {
		fprintf(stderr, "Failed to read FITS header from file \"%s\".\n", fn);
		return NULL;
	}
	result = readfunc(hdr, dest);
	if (!result) {
		fprintf(stderr, "Failed to parse WCS header from file \"%s\".\n", fn);
	}
	qfits_header_destroy(hdr);
	return result;
}

// silly little dispatch function to avoid casting - I like a modicum of type safety
static void* call_sip_read_header(const qfits_header* hdr, void* dest) {
	return sip_read_header(hdr, dest);
}
sip_t* sip_read_header_file(const char* fn, sip_t* dest) {
	return read_header_file(fn, dest, call_sip_read_header);
}

static void* call_tan_read_header(const qfits_header* hdr, void* dest) {
	return tan_read_header(hdr, dest);
}
tan_t* tan_read_header_file(const char* fn, tan_t* dest) {
	return read_header_file(fn, dest, call_tan_read_header);
}

static bool read_polynomial(const qfits_header* hdr, const char* format,
							int order, double* data, int datastride,
							bool skip_linear) {
	int i, j;
	char key[64];
	double nil = -HUGE_VAL;
	double val;
	for (i=0; i<=order; i++)
		for (j=0; (i+j)<=order; j++) {
			if (i+j < 1)
				continue;
			// FIXME - should we try to read it and not fail if it doesn't exist,
			// or not read it at all?  Is it reasonable for linear terms to exist
			// and be non-zero?
			if (skip_linear && (i+j < 2))
				continue;
			sprintf(key, format, i, j);
			val = qfits_header_getdouble(hdr, key, nil);
			if (val == nil) {
				fprintf(stderr, "SIP: warning: key \"%s\" not found; setting to zero.\n", key);
				val=0.0;
				//fprintf(stderr, "SIP: key \"%s\" not found.\n", key);
				//return FALSE;
			}
			data[i*datastride + j] = val;
		}
	return TRUE;
}

sip_t* sip_read_header(const qfits_header* hdr, sip_t* dest) {
	sip_t sip;
	char* str;
	const char* key;
	const char* expect;

	memset(&sip, 0, sizeof(sip_t));

	key = "CTYPE1";
	expect = "RA---TAN-SIP";
	str = qfits_header_getstr(hdr, key);
	str = qfits_pretty_string(str);
	if (!str) {
		fprintf(stderr, "SIP header: no %s.\n", key);
		return NULL;
	}
	if (strncmp(str, expect, strlen(expect))) {
		if (!tan_read_header(hdr, &(sip.wcstan))) {
			fprintf(stderr, "SIP: failed to read TAN header.\n");
			return NULL;
		}
		goto gohome;
	}

	key = "CTYPE2";
	expect = "DEC--TAN-SIP";
	str = qfits_header_getstr(hdr, key);
	str = qfits_pretty_string(str);
	if (!str || strncmp(str, expect, strlen(expect))) {
		fprintf(stderr, "SIP header: invalid \"%s\": expected \"%s\", got \"%s\".\n", key, expect, str);
		return NULL;
	}

	if (!tan_read_header(hdr, &sip.wcstan)) {
		fprintf(stderr, "SIP: failed to read TAN header.\n");
		return NULL;
	}

	sip.a_order  = qfits_header_getint(hdr, "A_ORDER", -1);
	sip.b_order  = qfits_header_getint(hdr, "B_ORDER", -1);
	sip.ap_order = qfits_header_getint(hdr, "AP_ORDER", -1);
	sip.bp_order = qfits_header_getint(hdr, "BP_ORDER", -1);

	if ((sip.a_order == -1) || 
		(sip.b_order == -1) || 
		(sip.ap_order == -1) || 
		(sip.bp_order == -1)) {
		fprintf(stderr, "SIP: failed to read polynomial orders.\n");
		return NULL;
	}

	if ((sip.a_order > SIP_MAXORDER) || 
		(sip.b_order > SIP_MAXORDER) || 
		(sip.ap_order > SIP_MAXORDER) || 
		(sip.bp_order > SIP_MAXORDER)) {
		fprintf(stderr, "SIP: polynomial orders (A=%i, B=%i, AP=%i, BP=%i) exceeds maximum of %i.\n",
				sip.a_order, sip.b_order, sip.ap_order, sip.bp_order, SIP_MAXORDER);
		return NULL;
	}

	if (!read_polynomial(hdr, "A_%i_%i",  sip.a_order,  (double*)sip.a,  SIP_MAXORDER, TRUE) ||
		!read_polynomial(hdr, "B_%i_%i",  sip.b_order,  (double*)sip.b,  SIP_MAXORDER, TRUE) ||
		!read_polynomial(hdr, "AP_%i_%i", sip.ap_order, (double*)sip.ap, SIP_MAXORDER, FALSE) ||
		!read_polynomial(hdr, "BP_%i_%i", sip.bp_order, (double*)sip.bp, SIP_MAXORDER, FALSE)) {
		fprintf(stderr, "SIP: failed to read polynomial terms.\n");
		return NULL;
	}

 gohome:
	if (!dest)
		dest = malloc(sizeof(sip_t));

	memcpy(dest, &sip, sizeof(sip_t));
	return dest;
}

tan_t* tan_read_header(const qfits_header* hdr, tan_t* dest) {
	char* str;
	const char* key;
	const char* expect;
	tan_t tan;
	double nil = -1e300;

	memset(&tan, 0, sizeof(tan_t));

	key = "CTYPE1";
	expect = "RA---TAN";
	str = qfits_header_getstr(hdr, key);
	str = qfits_pretty_string(str);
	if (!str || strncmp(str, expect, strlen(expect))) {
		fprintf(stderr, "TAN header: invalid \"%s\": expected \"%s\", got \"%s\".\n", key, expect, str);
		return NULL;
	}

	key = "CTYPE2";
	expect = "DEC--TAN";
	str = qfits_header_getstr(hdr, key);
	str = qfits_pretty_string(str);
	if (!str || strncmp(str, expect, strlen(expect))) {
		fprintf(stderr, "TAN header: invalid \"%s\": expected \"%s\", got \"%s\".\n", key, expect, str);
		return NULL;
	}

    tan.imagew = qfits_header_getint(hdr, "IMAGEW", 0);
    tan.imageh = qfits_header_getint(hdr, "IMAGEH", 0);

    if (!tan.imagew) {
        tan.imagew = qfits_header_getint(hdr, "NAXIS1", 0);
    }
    if (!tan.imageh) {
        tan.imageh = qfits_header_getint(hdr, "NAXIS2", 0);
    }

	{
		const char* keys[] = { "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
						 "CD1_1", "CD1_2", "CD2_1", "CD2_2" };
		double* vals[] = { &(tan.crval[0]), &(tan.crval[1]),
						   &(tan.crpix[0]), &(tan.crpix[1]),
						   &(tan.cd[0][0]), &(tan.cd[0][1]),
						   &(tan.cd[1][0]), &(tan.cd[1][1]) };
		int i;

		for (i=0; i<8; i++) {
			*(vals[i]) = qfits_header_getdouble(hdr, keys[i], nil);
			if (*(vals[i]) == nil) {
				fprintf(stderr, "TAN header: missing or invalid value for \"%s\".\n", keys[i]);
				return NULL;
			}
		}
	}

	if (!dest)
		dest = malloc(sizeof(tan_t));
	memcpy(dest, &tan, sizeof(tan_t));
	return dest;
}
