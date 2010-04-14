/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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

#include <stdio.h>

#ifdef WCSLIB_EXISTS
#include <wcshdr.h>
#include <wcs.h>
#endif

#include "anwcs.h"
#include "anqfits.h"
#include "qfits_std.h"
#include "errors.h"
#include "log.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "starutil.h"

struct anwcslib_t {
	struct wcsprm* wcs;
	// Image width and height, in pixels.
	int imagew;
	int imageh;
};
typedef struct anwcslib_t anwcslib_t;

// Approximate pixel scale, in arcsec/pixel, at the reference point.
double anwcs_pixel_scale(const anwcs_t* anwcs) {
	assert(anwcs);
	switch (anwcs->type) {

	case ANWCS_TYPE_WCSLIB:
		{
			anwcslib_t* anwcslib = anwcs->data;
			struct wcsprm* wcs = anwcslib->wcs;
			double* cd = wcs->m_cd;
			// HACK -- assume "cd" elements are set...
			return deg2arcsec(sqrt(fabs(cd[0]*cd[3] - cd[1]*cd[2])));
		}

	case ANWCS_TYPE_SIP:
		return sip_pixel_scale(anwcs->data);

	default:
		ERROR("Unknown anwcs type %i", anwcs->type);
		return -1;
	}
}

double anwcs_imagew(const anwcs_t* anwcs) {
	assert(anwcs);
	switch (anwcs->type) {

	case ANWCS_TYPE_WCSLIB:
		{
			anwcslib_t* anwcslib = anwcs->data;
			return anwcslib->imagew;
		}

	case ANWCS_TYPE_SIP:
		return sip_imagew(anwcs->data);

	default:
		ERROR("Unknown anwcs type %i", anwcs->type);
		return -1;
	}
}

double anwcs_imageh(const anwcs_t* anwcs) {
	assert(anwcs);
	switch (anwcs->type) {

	case ANWCS_TYPE_WCSLIB:
		{
			anwcslib_t* anwcslib = anwcs->data;
			return anwcslib->imageh;
		}
		
	case ANWCS_TYPE_SIP:
		return sip_imageh(anwcs->data);

	default:
		ERROR("Unknown anwcs type %i", anwcs->type);
		return -1;
	}
}

int anwcs_get_radec_center_and_radius(anwcs_t* anwcs,
									  double* p_ra, double* p_dec, double* p_radius) {
	assert(anwcs);
	switch (anwcs->type) {
	case ANWCS_TYPE_WCSLIB:
		{
			anwcslib_t* anwcslib = anwcs->data;
			double x,y;
			double ra1, dec1, ra2, dec2;
			// FIXME -- is this right?
			x = anwcslib->imagew + 0.5;
			y = anwcslib->imageh + 0.5;
			if (anwcs_pixelxy2radec(anwcs, x, y, &ra1, &dec1))
				return -1;
			// FIXME -- this is certainly not right in general....
			if (p_ra) *p_ra = ra1;
			if (p_dec) *p_dec = dec1;
			if (p_radius) {
				if (anwcs_pixelxy2radec(anwcs, 1.0, 1.0, &ra2, &dec2))
					return -1;
				*p_radius = deg_between_radecdeg(ra1, dec1, ra2, dec2);
			}
		}
		break;

	case ANWCS_TYPE_SIP:
		{
			sip_t* sip;
			sip = anwcs->data;
			sip_get_radec_center(sip, p_ra, p_dec);
			if (p_radius)
				*p_radius = sip_get_radius_deg(sip);
		}
		break;

	default:
		ERROR("Unknown anwcs type %i", anwcs->type);
		return -1;
	}
	return 0;
}

anwcs_t* anwcs_new_sip(const sip_t* sip) {
	anwcs_t* anwcs;
	anwcs = calloc(1, sizeof(anwcs_t));
	anwcs->type = ANWCS_TYPE_SIP;
	anwcs->data = sip_create();
	memcpy(anwcs->data, sip, sizeof(sip_t));
	return anwcs;
}

anwcs_t* anwcs_new_tan(const tan_t* tan) {
	sip_t sip;
	sip_wrap_tan(tan, &sip);
	return anwcs_new_sip(&sip);
}

anwcs_t* anwcs_open(const char* filename, int ext) {
	char* errmsg;
	anwcs_t* anwcs = NULL;
	errors_start_logging_to_string();

	// try as SIP:
	anwcs = anwcs_open_sip(filename, ext);
	if (anwcs) {
		errors_pop_state();
		return anwcs;
	} else {
		errmsg = errors_stop_logging_to_string(": ");
		logverb("Failed to open file %s, ext %i as SIP: %s", filename, ext, errmsg);
		free(errmsg);
	}

	// try as WCSLIB:
	anwcs = anwcs_open_wcslib(filename, ext);
	if (anwcs) {
		errors_pop_state();
		return anwcs;
	} else {
		errmsg = errors_stop_logging_to_string(": ");
		logverb("Failed to open file %s, ext %i using WCSLIB: %s", filename, ext, errmsg);
		free(errmsg);
	}

	return NULL;
}

anwcs_t* anwcs_open_sip(const char* filename, int ext) {
	anwcs_t* anwcs = NULL;
	sip_t* sip = NULL;
	sip = sip_read_tan_or_sip_header_file_ext(filename, ext, NULL, FALSE);
	if (!sip) {
		ERROR("Failed to parse SIP header");
		return NULL;
	}
	anwcs = calloc(1, sizeof(anwcs_t));
	anwcs->type = ANWCS_TYPE_SIP;
	anwcs->data = sip;
	return anwcs;
}

anwcs_t* anwcs_open_wcslib(const char* filename, int ext) {
#ifndef WCSLIB_EXISTS
	ERROR("Wcslib support was not compiled in");
	return NULL;
#else
	anqfits_t* fits = anqfits_open(filename);
	const qfits_header* hdr;
	char* hdrstr = NULL;
	int Nhdr;
	int nkeys;
	int nwcs = 0;
	int code;
	int nrej = 0;
	struct wcsprm* wcs = NULL;
	struct wcsprm* wcs2 = NULL;
	anwcs_t* anwcs = NULL;
	anwcslib_t* anwcslib;
	int W, H;

	if (!fits) {
		ERROR("Failed to open file %s", filename);
		return NULL;
	}
	hdrstr = anqfits_header_get_data(fits, ext, &Nhdr);
	hdr = anqfits_get_header_const(fits, ext);

	if (!hdrstr) {
		ERROR("Failed to read header data from file %s, ext %i", filename, ext);
		anqfits_close(fits);
		fits = NULL;
		return NULL;
	}
	nkeys = Nhdr / FITS_LINESZ;

	if (!sip_get_image_size(hdr, &W, &H)) {
		logverb("Failed to find image size in file %s, ext %i\n", filename, ext);
		W = H = 0;
	}

	anqfits_close(fits);
	fits = NULL;

	code = wcspih(hdrstr, nkeys,  WCSHDR_all, 2, &nrej, &nwcs, &wcs);
	free(hdrstr);
	hdrstr = NULL;
	if (code) {
		ERROR("wcslib's wcspih() failed with code %i", code);
		return NULL;
	}

	if (nwcs > 1) {
		// copy the first entry, free the rest.
		wcs2 = calloc(1, sizeof(struct wcsprm));
		wcscopy(1, wcs, wcs2);
		wcsvfree(&nwcs, &wcs);
	} else {
		wcs2 = wcs;
	}
	wcsset(wcs2);

	anwcs = calloc(1, sizeof(anwcs_t));
	anwcs->type = ANWCS_TYPE_WCSLIB;
	anwcs->data = calloc(1, sizeof(anwcslib_t));
	anwcslib = anwcs->data;
	anwcslib->wcs = wcs2;
	anwcslib->imagew = W;
	anwcslib->imageh = H;

	return anwcs;
#endif
}

int anwcs_radec2pixelxy(const anwcs_t* anwcs, double ra, double dec, double* px, double* py) {
	switch (anwcs->type) {
	case ANWCS_TYPE_WCSLIB:
#ifndef WCSLIB_EXISTS
	ERROR("Wcslib support was not compiled in");
	return -1;
#else
		{
			double pix[2];
			double world[2];
			double phi;
			double theta;
			double imgcrd[2];
			int status = 0;
			int code;
			anwcslib_t* anwcslib = anwcs->data;
			struct wcsprm* wcs = anwcslib->wcs;
			world[wcs->lng] = ra;
			world[wcs->lat] = dec;
			code = wcss2p(wcs, 1, 0, world, &phi, &theta, imgcrd, pix, &status);
			/*
			 int wcss2p(struct wcsprm *wcs, int ncoord, int nelem, const double world[],
			 double phi[], double theta[], double imgcrd[], double pixcrd[],
			 int stat[]);
			 */
			if (code) {
				ERROR("Wcslib's wcss2p() failed: code=%i, status=%i", code, status);
				return -1;
			}
			if (px) *px = pix[0];
			if (py) *py = pix[1];
		}
#endif
		break;

	case ANWCS_TYPE_SIP:
		{
			sip_t* sip;
			bool ok;
			sip = anwcs->data;
			ok = sip_radec2pixelxy(sip, ra, dec, px, py);
			if (!ok)
				return -1;
		}
		break;

	default:
		ERROR("Unknown anwcs type %i", anwcs->type);
		return -1;
	}
	return 0;
}

int anwcs_pixelxy2radec(const anwcs_t* anwcs, double px, double py, double* ra, double* dec) {
	assert(anwcs);
	switch (anwcs->type) {

	case ANWCS_TYPE_WCSLIB:
#ifndef WCSLIB_EXISTS
	ERROR("Wcslib support was not compiled in");
	return -1;
#else
		{
			double pix[2];
			double world[2];
			double phi;
			double theta;
			double imgcrd[2];
			int status = 0;
			int code;
			anwcslib_t* anwcslib = anwcs->data;
			struct wcsprm* wcs = anwcslib->wcs;
			pix[0] = px;
			pix[1] = py;
			code = wcsp2s(wcs, 1, 0, pix, imgcrd, &phi, &theta, world, &status);
			/*
			 int wcsp2s(struct wcsprm *wcs, int ncoord, int nelem, const double pixcrd[],
			 double imgcrd[], double phi[], double theta[], double world[],
			 int stat[]);
			 */
			if (code) {
				ERROR("Wcslib's wcsp2s() failed: code=%i, status=%i", code, status);
				return -1;
			}
			if (ra)  *ra  = world[wcs->lng];
			if (dec) *dec = world[wcs->lat];
		}
#endif
		break;

	case ANWCS_TYPE_SIP:
		sip_pixelxy2radec(anwcs->data, px, py, ra, dec);
		break;

	default:
		ERROR("Unknown anwcs type %i", anwcs->type);
		return -1;
	}
	return 0;
}

void anwcs_print(const anwcs_t* anwcs, FILE* fid) {
	assert(anwcs);
	assert(fid);
	switch (anwcs->type) {
	case ANWCS_TYPE_WCSLIB:
#ifndef WCSLIB_EXISTS
		fprintf(fid, "AN WCS type: wcslib, but wcslib support is not compiled in!\n");
		return;
#endif
		{
			anwcslib_t* anwcslib = anwcs->data;
			fprintf(fid, "AN WCS type: wcslib\n");
			wcsprt(anwcslib->wcs);
			fprintf(fid, "Image size: %i x %i\n", anwcslib->imagew, anwcslib->imageh);
			break;
		}

	case ANWCS_TYPE_SIP:
		fprintf(fid, "AN WCS type: sip\n");
		sip_print_to(anwcs->data, fid);
		break;

	default:
		fprintf(fid, "AN WCS type: unknown (%i)\n", anwcs->type);
	}
}

void anwcs_free(anwcs_t* anwcs) {
	if (!anwcs)
		return;
	switch (anwcs->type) {
	case ANWCS_TYPE_WCSLIB:
#ifdef WCSLIB_EXISTS
		{
			anwcslib_t* anwcslib = anwcs->data;
			wcsfree(anwcslib->wcs);
			free(anwcslib->wcs);
			free(anwcslib);
		}
#endif
		break;

	case ANWCS_TYPE_SIP:
		sip_free(anwcs->data);
		break;

	}
	free(anwcs);
}


