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

anwcs_t* anwcs_open_wcslib(const char* filename, int ext) {
#ifndef WCSLIB_EXISTS
	ERROR("Wcslib support was not compiled in");
	return NULL;
#else
	anqfits_t* fits = anqfits_open(filename);
	char* hdrstr = NULL;
	int Nhdr;
	int nkeys;
	int nwcs = 0;
	int code;
	int nrej = 0;
	struct wcsprm* wcs = NULL;
	struct wcsprm* wcs2 = NULL;
	anwcs_t* anwcs = NULL;

	if (!fits) {
		ERROR("Failed to open file %s", filename);
		return NULL;
	}
	hdrstr = anqfits_header_get_data(fits, ext, &Nhdr);
	anqfits_close(fits);
	fits = NULL;

	if (!hdrstr) {
		ERROR("Failed to read header data from file %s, ext %i", filename, ext);
		return NULL;
	}
	nkeys = Nhdr / FITS_LINESZ;

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
	anwcs->data = wcs2;

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
			struct wcsprm* wcs = anwcs->data;
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
			struct wcsprm* wcs = anwcs->data;
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
		fprintf(fid, "AN WCS type: wcslib\n");
		wcsprt(anwcs->data);
		break;
	default:
		fprintf(fid, "AN WCS type: unknown (%i)\n", anwcs->type);
	}
}

void anwcs_free(anwcs_t* wcs) {
	if (!wcs)
		return;
	switch (wcs->type) {
	case ANWCS_TYPE_WCSLIB:
#ifdef WCSLIB_EXISTS
		wcsfree(wcs->data);
		free(wcs->data);
#endif
		break;
	}
	free(wcs);
}


