
%module util

%include <typemaps.i>

%{
/*
#include "index.h"
#include "codekd.h"
#include "starkd.h"
#include "qidxfile.h"
*/

#include "log.h"
#include "healpix.h"
//#include "anwcs.h"
#include "sip.h"
#include "sip_qfits.h"

#define true 1
#define false 0

%}

// Things in keywords.h (used by healpix.h)
#define Const
#define WarnUnusedResult
#define ASTROMETRY_KEYWORDS_H

void log_init(int level);

//%apply double *OUTPUT { double *dx };
//%apply double *OUTPUT { double *dy };

%apply double *OUTPUT { double *dx, double *dy };
%apply double *OUTPUT { double *ra, double *dec };

%include "healpix.h"
//%include "anwcs.h"

%apply double *OUTPUT { double *p_x, double *p_y, double *p_z };
%apply double *OUTPUT { double *p_ra, double *p_dec };
//%apply double *OUTPUT { double *xyz };

%include "sip.h"
%include "sip_qfits.h"

%extend tan_t {
	tan_t(char* fn=NULL, int ext=0) {
		if (fn)
			return tan_read_header_file_ext(fn, ext, NULL);
		tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
		return t;
	}
	~tan_t() { free($self); }
	double pixel_scale() { return tan_pixel_scale($self); }
	void pixelxy2xyz(double x, double y, double *p_x, double *p_y, double *p_z) {
		double xyz[3];
		tan_pixelxy2xyzarr($self, x, y, xyz);
		*px = xyz[0];
		*py = xyz[1];
		*pz = xyz[2];
	}
	void pixelxy2radec(double x, double y, double *p_ra, double *p_dec) {
		tan_pixelxy2radec($self, x, y, p_ra, p_dec);
	}
	void radec2pixelxy(double ra, double dec, double *p_x, double *p_y) {
		tan_radec2pixelxy($self, ra, dec, p_x, p_y);
	}
	void xyz2pixelxy(double x, double y, double z, double *p_x, double *p_y) {
		double xyz[3];
		xyz[0] = x;
		xyz[1] = y;
		xyz[2] = z;
		tan_xyzarr2pixelxy($self, ra, dec, p_x, p_y);
	}


 };



%pythoncode %{
Tan = tan_t
%} 
