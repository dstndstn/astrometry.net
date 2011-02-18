
%module plotstuff_c
#undef ATTRIB_FORMAT
#define ATTRIB_FORMAT(x,y,z)
#undef WarnUnusedResult
#define WarnUnusedResult
%{
#include "numpy/arrayobject.h"
#include <sys/param.h>
#include <stdlib.h>

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
#include "coadd.h"
#include "qfits.h"
#include "mathutil.h"

#define true 1
#define false 0

	//#define Const
	//#define InlineDeclare

%}

%include "typemaps.i"

%include "plotstuff.h"
%include "coadd.h"
%include "qfits_image.h"
%include "fitsioutils.h"

 /*
  number* coadd_create_weight_image_from_range(const number* img, int W, int H,
  number lowval, number highval);
  */

void free(void* ptr);

%apply int* OUTPUT { int* newW, int* newH };
#define Const
#define InlineDeclare
%include "mathutil.h"
#undef Const
#undef InlineDeclare

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

%init %{
   import_array();
%}

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

%inline %{
	void image_debug(float* img, int W, int H) {
		int i;
		double mn,mx;
		mn = 1e300;
		mx = -1e300;
		for (i=0; i<(W*H); i++) {
			mn = MIN(mn, img[i]);
			mx = MAX(mx, img[i]);
		}
		logmsg("Image min,max %g,%g\n", mn,mx);
	}

	//void image_subtract_median(float* img, int W, int H) {
	//}

	void image_add(float* img, int W, int H, float val) {
		int i;
		for (i=0; i<(W*H); i++)
			img[i] += val;
	}


	%}


%extend plot_args {
	PyObject* get_image_as_numpy() {
		npy_intp dim[3];
		//PyObject* po;
		/*
		printf("get_image_as_numpy\n");
		printf("  image size %i x %i\n", self->W, self->H);
		printf("  image data: %p\n", self->outimage);
		 */
		dim[0] = self->H;
		dim[1] = self->W;
		dim[2] = 4;
		/*{
			int i;
			int acc = 0;
			unsigned char* ptr = self->outimage;
			for (i=0; i<(dim[0] * dim[1] * dim[2]); i++) {
				acc += (ptr[i] ? 1 : 0);
			}
			printf("acc %i\n", acc);
		 }*/
		return PyArray_SimpleNewFromData(3, dim, NPY_UBYTE, self->outimage);
		/*
		 po = PyArray_SimpleNewFromData(3, dim, NPY_UBYTE, self->outimage);
		 printf("po: %p\n", po);
		 printf("dim: %i\n", (int)PyArray_DIM(po, 0));
		 printf("dim: %i\n", (int)PyArray_DIM(po, 1));
		 printf("dim: %i\n", (int)PyArray_DIM(po, 2));
		 printf("itemsize: %i\n", PyArray_ITEMSIZE(po));
		 return po;
		 */
	}
}


%extend plotimage_args {

  int set_wcs_file(const char* fn, int ext) {
    return plot_image_set_wcs(self, fn, ext);
  }
  int set_file(const char* fn) {
    return plot_image_set_filename(self, fn);
  }

  void set_rgbscale(double r, double g, double b) {
	  self->rgbscale[0] = r;
	  self->rgbscale[1] = g;
	  self->rgbscale[2] = b;
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
