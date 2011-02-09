
%module util

%include <typemaps.i>

%{
/*
#include "index.h"
#include "codekd.h"
#include "starkd.h"
#include "qidxfile.h"
*/

// numpy.
#include <numpy/arrayobject.h>

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

/*
%typemap(in) double x, double y, double z (double xyz[3]) {
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
    $1 = xyz;
}
%typemap(in) double* xyz (double temp[3]) {
    temp[0] = xyz[0];
    temp[1] = xyz[1];
    temp[2] = xyz[2];
    $1 = temp;
}
 */


%typemap(in) double [ANY] (double temp[$1_dim0]) {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }
  if (PySequence_Length($input) != $1_dim0) {
    PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected $1_dim0 elements");
    return NULL;
  }
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyNumber_Check(o)) {
      temp[i] = PyFloat_AsDouble(o);
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");      
      return NULL;
    }
  }
  $1 = temp;
}
%typemap(out) double [ANY] {
  int i;
  $result = PyList_New($1_dim0);
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PyFloat_FromDouble($1[i]);
    PyList_SetItem($result,i,o);
  }
}

%typemap(in) double flatmatrix[ANY][ANY] (double temp[$1_dim0][$1_dim1]) {
    int i;
    if (!PySequence_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expected a sequence");
        return NULL;
    }
    if (PySequence_Length($input) != ($1_dim0 * $1_dim1)) {
        PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected $1_dim0*$1_dim1 elements");
        return NULL;
    }
    for (i = 0; i < ($1_dim0*$1_dim1); i++) {
        PyObject *o = PySequence_GetItem($input,i);
        if (PyNumber_Check(o)) {
            // FIXME -- is it dim0 or dim1?
            temp[i / $1_dim0][i % $1_dim0] = PyFloat_AsDouble(o);
        } else {
            PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");      
            return NULL;
        }
    }
    $1 = temp;
}
%typemap(out) double flatmatrix[ANY][ANY] {
  int i;
  $result = PyList_New($1_dim0 * $1_dim1);
  for (i = 0; i < ($1_dim0)*($1_dim1); i++) {
      // FIXME -- dim0 or dim1?
      PyObject *o = PyFloat_FromDouble($1[i / $1_dim0][i % $1_dim0]);
      PyList_SetItem($result,i,o);
  }
 }


%apply double [ANY] { double crval[2] };
%apply double [ANY] { double crpix[2] };
%apply double flatmatrix[ANY][ANY] { double cd[2][2] };




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
        *p_x = xyz[0];
        *p_y = xyz[1];
        *p_z = xyz[2];
    }
    void pixelxy2radec(double x, double y, double *p_ra, double *p_dec) {
        tan_pixelxy2radec($self, x, y, p_ra, p_dec);
    }
    int radec2pixelxy(double ra, double dec, double *p_x, double *p_y) {
        return tan_radec2pixelxy($self, ra, dec, p_x, p_y);
    }
    int xyz2pixelxy(double x, double y, double z, double *p_x, double *p_y) {
        double xyz[3];
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        return tan_xyzarr2pixelxy($self, xyz, p_x, p_y);
    }
 };


%inline %{

	static int tan_numpy_pixelxy2radec(tan_t* tan, PyObject* npx, PyObject* npy, 
									   PyObject* npra, PyObject* npdec) {
										
		int i, N;
		double *x, *y, *ra, *dec;
		
		if (PyArray_NDIM(npx) != 1) {
			PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional");
			return -1;
		}
		if (PyArray_TYPE(npx) != PyArray_DOUBLE) {
			PyErr_SetString(PyExc_ValueError, "array must contain doubles");
			return -1;
		}
		N = PyArray_DIM(npx, 0);
		if ((PyArray_DIM(npy, 0) != N) ||
			(PyArray_DIM(npra, 0) != N) ||
			(PyArray_DIM(npdec, 0) != N)) {
			PyErr_SetString(PyExc_ValueError, "arrays must be the same size");
			return -1;
		}
		x = PyArray_GETPTR1(npx, 0);
		y = PyArray_GETPTR1(npy, 0);
		ra = PyArray_GETPTR1(npra, 0);
		dec = PyArray_GETPTR1(npdec, 0);
		for (i=0; i<N; i++)
			tan_pixelxy2radec(tan, x[i], y[i], ra+i, dec+i);
		return 0;
	}

%}

%pythoncode %{
import numpy as np

def tan_t_tostring(self):
    return ('Tan: crpix (%.1f, %.1f), crval (%g, %g), cd (%g, %g, %g, %g), image %g x %g' %
            (self.crpix[0], self.crpix[1], self.crval[0], self.crval[1],
             self.cd[0], self.cd[1], self.cd[2], self.cd[3],
             self.imagew, self.imageh))
tan_t.__str__ = tan_t_tostring

def tan_t_pixelxy2radec_any(self, x, y):
    if np.iterable(x) or np.iterable(y):
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)
        r = np.empty(len(x))
        d = np.empty(len(x))
        print 'self:', self
        print 'self.this:', self.this
        print 'x', x
        print type(x)
        print 'y', y
        print 'r', r
        print 'd', d
        tan_numpy_pixelxy2radec(self.this, x, y, r, d)
        return r,d
    else:
        return self.pixelxy2radec_single(x, y)
tan_t.pixelxy2radec_single = tan_t.pixelxy2radec
tan_t.pixelxy2radec = tan_t_pixelxy2radec_any



Tan = tan_t
%} 
