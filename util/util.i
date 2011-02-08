
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

 //%nodefaultctor tan_t;
 //%rename(Tan) tan_t;
//%rename(Sip) sip_t;
%include "sip.h"
%include "sip_qfits.h"

%apply double *OUTPUT { double *px, double *py, double *pz };

%inline %{
	//typedef tan_t Tan;
 %}

%extend tan_t {
	tan_t(char* fn=NULL, int ext=0) {
		if (fn)
			return tan_read_header_file_ext(fn, ext, NULL);
		tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
		return t;
	}
	~tan_t() { free($self); }
	double pixel_scale() { return tan_pixel_scale($self); }
	void pixelxy2xyz(double x, double y, double *px, double *py, double *pz) {
		double xyz[3];
		tan_pixelxy2xyzarr($self, x, y, xyz);
		*px = xyz[0];
		*py = xyz[1];
		*pz = xyz[2];
	}
 };


%pythoncode %{
Tan = tan_t
%} 
