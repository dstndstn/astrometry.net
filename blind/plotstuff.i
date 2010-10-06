
%module plotstuff_c
#undef ATTRIB_FORMAT
#define ATTRIB_FORMAT(x,y,z)
#undef WarnUnusedResult
#define WarnUnusedResult
%{
#include "plotstuff.h"
#include "plotimage.h"
#include "plotoutline.h"
#include "plotgrid.h"
#include "plotindex.h"
#include "plotxy.h"
#include "plotradec.h"
#include "plotmatch.h"
#include "plotannotations.h"
#include "sip.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "log.h"
#include "fitsioutils.h"
#include "anwcs.h"
#define true 1
#define false 0
%}
%include "plotstuff.h"
%include "plotimage.h"
%include "plotoutline.h"
%include "plotgrid.h"
%include "plotindex.h"
%include "plotxy.h"
%include "plotradec.h"
%include "plotmatch.h"
%include "plotannotations.h"
%include "sip.h"
%include "sip_qfits.h"
%include "sip-utils.h"
%include "anwcs.h"

enum log_level {
	LOG_NONE,
	LOG_ERROR,
	LOG_MSG,
	LOG_VERB,
	LOG_ALL
};

// HACK!
enum cairo_op {
    CAIRO_OPERATOR_CLEAR,
    CAIRO_OPERATOR_SOURCE,
    CAIRO_OPERATOR_OVER,
    CAIRO_OPERATOR_IN,
    CAIRO_OPERATOR_OUT,
    CAIRO_OPERATOR_ATOP,
    CAIRO_OPERATOR_DEST,
    CAIRO_OPERATOR_DEST_OVER,
    CAIRO_OPERATOR_DEST_IN,
    CAIRO_OPERATOR_DEST_OUT,
    CAIRO_OPERATOR_DEST_ATOP,
    CAIRO_OPERATOR_XOR,
    CAIRO_OPERATOR_ADD,
    CAIRO_OPERATOR_SATURATE
};
typedef enum cairo_op cairo_operator_t;

void log_init(int log_level);
void fits_use_error_system(void);

%extend sip_t {
	double crval1() {
		return self->wcstan.crval[0];
	}
	double crval2() {
		return self->wcstan.crval[1];
	}
}

%extend plotimage_args {
  int set_wcs_file(const char* fn, int ext) {
    return plot_image_set_wcs(self, fn, ext);
  }
  int set_file(const char* fn) {
    return plot_image_set_filename(self, fn);
  }

  int get_image_width() {
	  int W;
	  if (plot_image_getsize(self, &W, NULL)) {
		  return -1;
	  }
	  return W;
  }
  int get_image_height() {
	  int H;
	  if (plot_image_getsize(self, NULL, &H)) {
		  return -1;
	  }
	  return H;
  }

}

%extend plotindex_args {
 int add_file(const char* fn) {
  return plot_index_add_file(self, fn);
}
}
