
%module plotstuff_c
#undef ATTRIB_FORMAT
#define ATTRIB_FORMAT(x,y,z)
#undef WarnUnusedResult
#define WarnUnusedResult
%{
#include "plotstuff.h"
#include "plotoutline.h"
#include "plotgrid.h"
#include "plotindex.h"
#include "sip.h"
#include "log.h"
#define true 1
#define false 0
%}
%include "plotstuff.h"
%include "plotoutline.h"
%include "plotgrid.h"
%include "plotindex.h"
%include "sip.h"
extern void log_init(int level);

%extend sip_t {
	double crval1() {
		return self->wcstan.crval[0];
	}
	double crval2() {
		return self->wcstan.crval[1];
	}
}

