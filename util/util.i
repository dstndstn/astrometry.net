
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
%include "sip.h"
