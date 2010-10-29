
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

%include "typemaps.i"

%include "plotstuff.h"

/*
	%apply unsigned char* rgboutdouble *OUTPUT { double *result };
	%inlne %{
	extern void add(double a, double b, double *result);
	%}
*/


/* Set the input argument to point to a temporary variable */
%typemap(in, numinputs=0) unsigned char* rgbout (unsigned char temp[3]) {
   $1 = temp;
}

%typemap(argout) unsigned char* rgbout {
  // Append output value $1 to $result
  if (result) {
    Py_DECREF($result);
    $result = Py_None;
  } else {
    int i;
    Py_DECREF($result);
    $result = PyList_New(3);
    for (i=0; i<3; i++) {
      PyObject *o = PyInt_FromLong((long)$1[i]);
      PyList_SetItem($result,i,o);
    }
  }
}

%typemap(in) int rgb[3] (int temp[3]) {
  int i;
  // Convert sequence of ints to int[3]
  if (!PySequence_Check($input) ||	 
  	 (PySequence_Length($input) != 3)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence of length 3");
    return NULL;
  }
  for (i=0; i<3; i++) {
    PyObject *o = PySequence_GetItem($input, i);
    if (PyNumber_Check(o)) {
      temp[i] = (int)PyInt_AsLong(o);
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");
      return NULL;
    }
  }
  $1 = temp;
}

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
