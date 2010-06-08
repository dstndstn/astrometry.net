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
#include <string.h>

#include "sip_qfits.h"
#include "an-bool.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"

sip_t* sip_read_tan_or_sip_header_file_ext(const char* wcsfn, int ext, sip_t* dest, bool forcetan) {
	sip_t* rtn;
	if (forcetan) {
		sip_t sip;
		memset(&sip, 0, sizeof(sip_t));
		if (!tan_read_header_file_ext(wcsfn, ext, &(sip.wcstan))) {
			ERROR("Failed to parse TAN header from file %s, extension %i", wcsfn, ext);
			return NULL;
		}
		if (!dest)
			dest = malloc(sizeof(sip_t));
		memcpy(dest, &sip, sizeof(sip_t));
		return dest;
	} else {
		rtn = sip_read_header_file_ext(wcsfn, ext, dest);
		if (!rtn)
			ERROR("Failed to parse SIP header from file %s, extension %i", wcsfn, ext);
		return rtn;
	}
}

int sip_write_to(const sip_t* sip, FILE* fid) {
	qfits_header* hdr;
	int res;
	hdr = sip_create_header(sip);
	if (!hdr) {
		ERROR("Failed to create FITS header from WCS");
		return -1;
	}
	res = qfits_header_dump(hdr, fid);
	qfits_header_destroy(hdr);
	return res;
}

int sip_write_to_file(const sip_t* sip, const char* fn) {
	FILE* fid;
	int res;
	fid = fopen(fn, "wb");
	if (!fid) {
		SYSERROR("Failed to open file \"%s\" to write WCS header", fn);
		return -1;
	}
	res = sip_write_to(sip, fid);
	if (res) {
		ERROR("Failed to write FITS header to file \"%s\"", fn);
		return -1;
	}
	if (fclose(fid)) {
		SYSERROR("Failed to close file \"%s\" after writing WCS header", fn);
		return -1;
	}
	return 0;
}

int tan_write_to(const tan_t* tan, FILE* fid) {
	qfits_header* hdr;
	int res;
	hdr = tan_create_header(tan);
	if (!hdr) {
		ERROR("Failed to create FITS header from WCS");
		return -1;
	}
	res = qfits_header_dump(hdr, fid);
	qfits_header_destroy(hdr);
	return res;
}

int tan_write_to_file(const tan_t* tan, const char* fn) {
	FILE* fid;
	int res;
	fid = fopen(fn, "wb");
	if (!fid) {
		SYSERROR("Failed to open file \"%s\" to write WCS header", fn);
		return -1;
	}
	res = tan_write_to(tan, fid);
	if (res) {
		ERROR("Failed to write FITS header to file \"%s\"", fn);
		return -1;
	}
	if (fclose(fid)) {
		SYSERROR("Failed to close file \"%s\" after writing WCS header", fn);
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

int sip_get_image_size(const qfits_header* hdr, int* pW, int* pH) {
	int W, H;
    W = qfits_header_getint(hdr, "IMAGEW", 0);
	debug("sip_get_image_size: IMAGEW = %i\n", W);
    H = qfits_header_getint(hdr, "IMAGEH", 0);
	debug("sip_get_image_size: IMAGEH = %i\n", H);
    if (!W) {
        W = qfits_header_getint(hdr, "NAXIS1", 0);
		debug("sip_get_image_size: NAXIS1 = %i\n", W);
	}
    if (!H) {
        H = qfits_header_getint(hdr, "NAXIS2", 0);
		debug("sip_get_image_size: NAXIS2 = %i\n", H);
	}
	if (pW) *pW = W;
	if (pH) *pH = H;
	return 0;
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

static void* read_header_file(const char* fn, int ext, void* dest,
							  void* (*readfunc)(const qfits_header*, void*)) {
	qfits_header* hdr;
	void* result;
	hdr = qfits_header_readext(fn, ext);
	if (!hdr) {
		ERROR("Failed to read FITS header from file \"%s\" extension %i", fn, ext);
		return NULL;
	}
	result = readfunc(hdr, dest);
	if (!result) {
		ERROR("Failed to parse WCS header from file \"%s\" extension %i", fn, ext);
	}
	qfits_header_destroy(hdr);
	return result;
}

// silly little dispatch function to avoid casting - I like a modicum of type safety
static void* call_sip_read_header(const qfits_header* hdr, void* dest) {
	return sip_read_header(hdr, dest);
}
sip_t* sip_read_header_file(const char* fn, sip_t* dest) {
	return read_header_file(fn, 0, dest, call_sip_read_header);
}
sip_t* sip_read_header_file_ext(const char* fn, int ext, sip_t* dest) {
	return read_header_file(fn, ext, dest, call_sip_read_header);
}

static void* call_tan_read_header(const qfits_header* hdr, void* dest) {
	return tan_read_header(hdr, dest);
}
tan_t* tan_read_header_file(const char* fn, tan_t* dest) {
	return read_header_file(fn, 0, dest, call_tan_read_header);
}
tan_t* tan_read_header_file_ext(const char* fn, int ext, tan_t* dest) {
	return read_header_file(fn, ext, dest, call_tan_read_header);
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
                ERROR("SIP: warning: key \"%s\" not found; setting to zero.", key);
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
		ERROR("SIP header: no key \"%s\"", key);
		return NULL;
	}
	if (strncmp(str, expect, strlen(expect))) {
		if (!tan_read_header(hdr, &(sip.wcstan))) {
			ERROR("SIP: failed to read TAN header");
			return NULL;
		}
		goto gohome;
	}

	key = "CTYPE2";
	expect = "DEC--TAN-SIP";
	str = qfits_header_getstr(hdr, key);
	str = qfits_pretty_string(str);
	if (!str || strncmp(str, expect, strlen(expect))) {
		ERROR("SIP header: incorrect key \"%s\": expected \"%s\", got \"%s\"", key, expect, str);
		return NULL;
	}

	if (!tan_read_header(hdr, &sip.wcstan)) {
		ERROR("SIP: failed to read TAN header");
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
		ERROR("SIP: failed to read polynomial orders");
		return NULL;
	}

	if ((sip.a_order > SIP_MAXORDER) || 
		(sip.b_order > SIP_MAXORDER) || 
		(sip.ap_order > SIP_MAXORDER) || 
		(sip.bp_order > SIP_MAXORDER)) {
		ERROR("SIP: polynomial orders (A=%i, B=%i, AP=%i, BP=%i) exceeds maximum of %i",
              sip.a_order, sip.b_order, sip.ap_order, sip.bp_order, SIP_MAXORDER);
		return NULL;
	}

	if (!read_polynomial(hdr, "A_%i_%i",  sip.a_order,  (double*)sip.a,  SIP_MAXORDER, TRUE) ||
		!read_polynomial(hdr, "B_%i_%i",  sip.b_order,  (double*)sip.b,  SIP_MAXORDER, TRUE) ||
		!read_polynomial(hdr, "AP_%i_%i", sip.ap_order, (double*)sip.ap, SIP_MAXORDER, FALSE) ||
		!read_polynomial(hdr, "BP_%i_%i", sip.bp_order, (double*)sip.bp, SIP_MAXORDER, FALSE)) {
		ERROR("SIP: failed to read polynomial terms");
		return NULL;
	}

 gohome:
	if (!dest)
		dest = malloc(sizeof(sip_t));

	memcpy(dest, &sip, sizeof(sip_t));
	return dest;
}

static int check_tan_ctypes(char* ct1, char* ct2) {
	const char* ra = "RA---TAN";
	const char* dec = "DEC--TAN";
	int NC = 8;
	if (!ct1 || !ct2)
		return -1;
	if (strlen(ct1) < NC || strlen(ct2) < NC)
		return -1;
	if ((strncmp(ct1, ra, NC) == 0) && (strncmp(ct2, dec, NC) == 0))
		return 0;
	if ((strncmp(ct1, dec, NC) == 0) && (strncmp(ct2, ra, NC) == 0))
		return 1;
	return -1;
}

tan_t* tan_read_header(const qfits_header* hdr, tan_t* dest) {
	tan_t tan;
	double nil = -1e300;
	char* ct1;
	char* ct2;
	int swap;
	int W, H;

	memset(&tan, 0, sizeof(tan_t));

	ct1 = fits_get_dupstring(hdr, "CTYPE1");
	ct2 = fits_get_dupstring(hdr, "CTYPE2");
	swap = check_tan_ctypes(ct1, ct2);
	if (swap == -1) {
		ERROR("TAN header: expected CTYPE1 = RA---TAN, CTYPE2 = DEC--TAN (or vice versa), get CTYPE1 = \"%s\", CYTPE2 = \"%s\"\n",
			  ct1, ct2);
	}
	free(ct1);
	free(ct2);
	if (swap == -1)
		return NULL;

	sip_get_image_size(hdr, &W, &H);
	tan.imagew = W;
	tan.imageh = H;

	{
		const char* keys[] = { "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
						 "CD1_1", "CD1_2", "CD2_1", "CD2_2" };
		double* vals[] = { &(tan.crval[0]), &(tan.crval[1]),
						   &(tan.crpix[0]), &(tan.crpix[1]),
						   &(tan.cd[0][0]), &(tan.cd[0][1]),
						   &(tan.cd[1][0]), &(tan.cd[1][1]) };
		int i;

		for (i=0; i<sizeof(keys)/sizeof(char*); i++) {
			*(vals[i]) = qfits_header_getdouble(hdr, keys[i], nil);
			if (*(vals[i]) == nil) {
				ERROR("TAN header: missing or invalid value for \"%s\"", keys[i]);
				return NULL;
			}
		}
	}

	if (swap == 1) {
		double tmp;
		tmp = tan.crval[0];
		tan.crval[0] = tan.crval[1];
		tan.crval[1] = tmp;
		// swap CD1_1 <-> CD2_1
		tmp = tan.cd[0][0];
		tan.cd[0][0] = tan.cd[1][0];
		tan.cd[1][0] = tmp;
		// swap CD1_2 <-> CD2_2
		tmp = tan.cd[0][1];
		tan.cd[0][1] = tan.cd[1][1];
		tan.cd[1][1] = tmp;
	}

	if (!dest)
		dest = malloc(sizeof(tan_t));
	memcpy(dest, &tan, sizeof(tan_t));
	return dest;
}
