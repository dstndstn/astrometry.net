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
#include <sys/param.h>

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

/*
 This is ugly... this macro gets defined differently depending on
 whether wcslib is available or not... I couldn't figure out how to put
 the #ifdef inside the macro definition to make it cleaner.
 */

#ifdef WCSLIB_EXISTS

#define ANWCS_DISPATCH(anwcs, action, defaction, func, ...)	\
	do {													\
		assert(anwcs);										\
		switch (anwcs->type) {								\
		case ANWCS_TYPE_WCSLIB:											\
			{															\
				anwcslib_t* anwcslib = anwcs->data;						\
				action wcslib_##func(anwcslib, ##__VA_ARGS__);			\
				break;													\
			}															\
		case ANWCS_TYPE_SIP:											\
			{															\
				sip_t* sip = anwcs->data;								\
				action ansip_##func(sip, ##__VA_ARGS__);				\
				break;													\
			}															\
		default:														\
			ERROR("Unknown anwcs type %i", anwcs->type);				\
			defaction;													\
		}																\
	} while (0)

#else

// No WCSLIB.
#define ANWCS_DISPATCH(anwcs, action, defaction, func, ...)	\
	do {													\
		assert(anwcs);										\
		switch (anwcs->type) {								\
		case ANWCS_TYPE_WCSLIB:									\
			ERROR("Wcslib support was not compiled in");				\
			defaction;													\
			break;														\
		case ANWCS_TYPE_SIP:											\
			{															\
				sip_t* sip = anwcs->data;								\
				action ansip_##func(sip, ##__VA_ARGS__);				\
				break;													\
			}															\
		default:														\
			ERROR("Unknown anwcs type %i", anwcs->type);				\
			defaction;													\
		}																\
	} while (0)

#endif



/////////////////// wcslib implementations //////////////////////////

#ifdef WCSLIB_EXISTS

static double wcslib_imagew(const anwcslib_t* anwcs) {
	return anwcs->imagew;
}
static double wcslib_imageh(const anwcslib_t* anwcs) {
	return anwcs->imageh;
}

static int wcslib_pixelxy2radec(const anwcslib_t* anwcslib, double px, double py, double* ra, double* dec) {
	double pix[2];
	double world[2];
	double phi;
	double theta;
	double imgcrd[2];
	int status = 0;
	int code;
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
	return 0;
}

static int wcslib_radec2pixelxy(const anwcslib_t* anwcslib, double ra, double dec, double* px, double* py) {
	double pix[2];
	double world[2];
	double phi;
	double theta;
	double imgcrd[2];
	int status = 0;
	int code;
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
	return 0;
}

static bool wcslib_radec_is_inside_image(anwcslib_t* wcslib, double ra, double dec) {
	double px, py;
	if (wcslib_radec2pixelxy(wcslib, ra, dec, &px, &py))
		return FALSE;
	return (px >= 1 && px <= wcslib->imagew &&
			py >= 1 && py <= wcslib->imageh);
}

static void wcslib_radec_bounds(const anwcslib_t* wcs, int stepsize,
								double* pramin, double* pramax,
								double* pdecmin, double* pdecmax) {
	// FIXME -- this does no pole-checking.
	// It should be possible to just walk the boundary (as in sip-utils)
	// but this is more robust...
	int i, j, W, H;
	double ralo, rahi, declo, dechi;
	// for gcc:
	ralo = rahi = declo = dechi = 0;
	W = wcslib_imagew(wcs);
	H = wcslib_imageh(wcs);
	for (i=1; i<H+stepsize; i+=stepsize) {
		// up to and including H
		i = MIN(H, i);
		for (j=1; j<=W+stepsize; j+=stepsize) {
			j = MIN(W, j);
			double ra,dec;
			if (wcslib_pixelxy2radec(wcs, i, j, &ra, &dec)) {
				ERROR("Error converting pixel coord (%i,%i) in radec_bounds", i, j);
				continue;
			}
			// first time through loop?
			if (i == 1 && j == 1) {
				ralo = rahi = ra;
				declo = dechi = dec;
			} else {
				ralo = MIN(ralo, ra);
				rahi = MAX(rahi, ra);
				declo = MIN(declo, dec);
				dechi = MAX(dechi, dec);
			}
		}
	}
	if (pramin) *pramin = ralo;
	if (pramax) *pramax = rahi;
	if (pdecmin) *pdecmin = declo;
	if (pdecmax) *pdecmax = dechi;
}

static void wcslib_print(const anwcslib_t* anwcslib, FILE* fid) {
	fprintf(fid, "AN WCS type: wcslib\n");
	wcsprt(anwcslib->wcs);
	fprintf(fid, "Image size: %i x %i\n", anwcslib->imagew, anwcslib->imageh);
}

static void wcslib_free(anwcslib_t* anwcslib) {
	wcsfree(anwcslib->wcs);
	free(anwcslib->wcs);
	free(anwcslib);
}

static double wcslib_pixel_scale(const anwcslib_t* anwcslib) {
	struct wcsprm* wcs = anwcslib->wcs;
	double* cd = wcs->m_cd;
	// HACK -- assume "cd" elements are set...
	return deg2arcsec(sqrt(fabs(cd[0]*cd[3] - cd[1]*cd[2])));
}

static int wcslib_write_to(const anwcslib_t* anwcslib, FILE* fid) {
	int res;
	int Ncards;
	char* hdrstr;
	res = wcshdo(-1, anwcslib->wcs, &Ncards, &hdrstr);
	if (res) {
		ERROR("wcshdo() failed: %s", wcshdr_errmsg[res]);
		return -1;
	}

	// FIXME -- incomplete!  Add other required headers and write to file.
	int i;
	printf("wcslib header:\n");
	for (i=0; i<Ncards; i++)
		printf("%.80s\n", hdrstr + i*80);
	printf("\n\n");

	ERROR("wcslib_write_to() is unfinished.\n");
	return -1;
}

static int wcslib_write(const anwcslib_t* anwcslib, const char* filename) {
	int rtn;
	FILE* fid = fopen(filename, "wb");
	if (!fid) {
		SYSERROR("Failed to open file \"%s\" for FITS WCS output", filename);
		return -1;
	}
	rtn = wcslib_write_to(anwcslib, fid);
	if (fclose(fid)) {
		if (!rtn) {
			SYSERROR("Failed to close output file \"%s\"", filename);
			return -1;
		}
	}
	if (rtn) {
		ERROR("wcslib_write_to file \"%s\" failed", filename);
		return -1;
	}
	return 0;
}

static void wcslib_set_size(anwcslib_t* anwcslib, int W, int H) {
	anwcslib->imagew = W;
	anwcslib->imageh = H;
}



#endif  // end of WCSLIB implementations


/////////////////// sip implementations //////////////////////////

/*
 static void ansip_radec_bounds(const sip_t* sip, int stepsize,
 double* pramin, double* pramax,
 double* pdecmin, double* pdecmax) {
 sip_get_radec_bounds(sip, stepsize, pramin, pramax, pdecmin, pdecmax);
 }
 */
#define ansip_radec_bounds sip_get_radec_bounds

#define ansip_radec_is_inside_image sip_is_inside_image

#define ansip_imagew sip_imagew
#define ansip_imageh sip_imageh

//#define ansip_pixelxy2radec sip_pixelxy2radec
static int ansip_pixelxy2radec(const sip_t* sip, double px, double py, double* ra, double* dec) {
	sip_pixelxy2radec(sip, px, py, ra, dec);
	return 0;
}

#define ansip_print sip_print_to

#define ansip_free sip_free

#define ansip_pixel_scale sip_pixel_scale

#define ansip_write sip_write_to_file

#define ansip_write_to sip_write_to

static void ansip_set_size(sip_t* sip, int W, int H) {
	sip->wcstan.imagew = W;
	sip->wcstan.imageh = H;
}

/////////////////// dispatched anwcs_t entry points //////////////////////////

void anwcs_set_size(anwcs_t* wcs, int W, int H) {
	ANWCS_DISPATCH(wcs, , , set_size, W, H);
}

void anwcs_get_radec_bounds(const anwcs_t* wcs, int stepsize,
							double* pramin, double* pramax,
							double* pdecmin, double* pdecmax) {
	ANWCS_DISPATCH(wcs, , , radec_bounds, stepsize, pramin, pramax, pdecmin, pdecmax);
}

void anwcs_print(const anwcs_t* anwcs, FILE* fid) {
	assert(anwcs);
	assert(fid);
	ANWCS_DISPATCH(anwcs, , , print, fid);
}

void anwcs_free(anwcs_t* anwcs) {
	if (!anwcs)
		return;
	ANWCS_DISPATCH(anwcs, , , free);
	free(anwcs);
}

bool anwcs_radec_is_inside_image(const anwcs_t* wcs, double ra, double dec) {
	ANWCS_DISPATCH(wcs, return, return FALSE, radec_is_inside_image, ra, dec);
}

double anwcs_imagew(const anwcs_t* anwcs) {
	ANWCS_DISPATCH(anwcs, return, return -1.0, imagew);
}
double anwcs_imageh(const anwcs_t* anwcs) {
	ANWCS_DISPATCH(anwcs, return, return -1.0, imageh);
}

int anwcs_pixelxy2radec(const anwcs_t* anwcs, double px, double py, double* ra, double* dec) {
	ANWCS_DISPATCH(anwcs, return, return -1, pixelxy2radec, px, py, ra, dec);
}

// Approximate pixel scale, in arcsec/pixel, at the reference point.
double anwcs_pixel_scale(const anwcs_t* anwcs) {
	ANWCS_DISPATCH(anwcs, return, return -1, pixel_scale);
}

int anwcs_write(const anwcs_t* wcs, const char* filename) {
	ANWCS_DISPATCH(wcs, return, return -1, write, filename);
}

int anwcs_write_to(const anwcs_t* wcs, FILE* fid) {
	ANWCS_DISPATCH(wcs, return, return -1, write_to, fid);
}







///////////////////////// un-dispatched functions ///////////////////

void anwcs_walk_image_boundary(const anwcs_t* wcs, double stepsize,
							   void (*callback)(const anwcs_t* wcs, double x, double y, double ra, double dec, void* token),
							   void* token) {
    int i, side;
    // Walk the perimeter of the image in steps of stepsize pixels
    double W = anwcs_imagew(wcs);
    double H = anwcs_imageh(wcs);
	logverb("Walking WCS image boundary: image size is %g x %g\n", W, H);
    {
		double Xmin = 0.5;
		double Xmax = W + 0.5;
		double Ymin = 0.5;
		double Ymax = H + 0.5;
        double offsetx[] = { Xmin, Xmax, Xmax, Xmin };
        double offsety[] = { Ymin, Ymin, Ymax, Ymax };
        double stepx[] = { +stepsize, 0, -stepsize, 0 };
        double stepy[] = { 0, +stepsize, 0, -stepsize };
        int Nsteps[] = { ceil(W/stepsize), ceil(H/stepsize), ceil(W/stepsize), ceil(H/stepsize) };

        for (side=0; side<4; side++) {
            for (i=0; i<Nsteps[side]; i++) {
                double ra, dec;
                double x, y;
                x = MIN(Xmax, MAX(Xmin, offsetx[side] + i * stepx[side]));
                y = MIN(Ymax, MAX(Ymin, offsety[side] + i * stepy[side]));
                anwcs_pixelxy2radec(wcs, x, y, &ra, &dec);
				callback(wcs, x, y, ra, dec, token);
            }
        }
    }
}


// FIXME -- this is probably the bass-ackwards way -- xyz is more natural; this probably requires converting back and forth between ra,dec and xyz.
int anwcs_pixelxy2xyz(const anwcs_t* wcs, double px, double py, double* xyz) {
	int rtn;
	double ra,dec;
	rtn = anwcs_pixelxy2radec(wcs, px, py, &ra, &dec);
	radecdeg2xyzarr(ra, dec, xyz);
	return rtn;
}

int anwcs_xyz2pixelxy(const anwcs_t* wcs, const double* xyz, double *px, double *py) {
	int rtn;
	double ra,dec;
	xyzarr2radecdeg(xyz, &ra, &dec);
	rtn = anwcs_radec2pixelxy(wcs, ra, dec, px, py);
	return rtn;
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
		errmsg = errors_stop_logging_to_string("\n  ");
		logverb("Failed to open file %s, ext %i as SIP:\n%s\n", filename, ext, errmsg);
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

static anwcs_t* open_tansip(const char* filename, int ext, bool forcetan) {
	anwcs_t* anwcs = NULL;
	sip_t* sip = NULL;
	sip = sip_read_tan_or_sip_header_file_ext(filename, ext, NULL, forcetan);
	if (!sip) {
		ERROR("Failed to parse SIP header");
		return NULL;
	}
	anwcs = calloc(1, sizeof(anwcs_t));
	anwcs->type = ANWCS_TYPE_SIP;
	anwcs->data = sip;
	return anwcs;
}

anwcs_t* anwcs_open_tan(const char* filename, int ext) {
	return open_tansip(filename, ext, TRUE);
}

anwcs_t* anwcs_open_sip(const char* filename, int ext) {
	return open_tansip(filename, ext, FALSE);
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

	if (sip_get_image_size(hdr, &W, &H)) {
		logverb("Failed to find image size in file %s, ext %i\n", filename, ext);
		//logverb("Header:\n");
		//qfits_header_debug_dump(hdr);
		logverb("\n");
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
			anwcslib_t* anwcslib = anwcs->data;
			return wcslib_radec2pixelxy(anwcslib, ra, dec, px, py);
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



