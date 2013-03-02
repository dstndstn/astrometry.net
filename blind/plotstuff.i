
%module(package="astrometry.blind") plotstuff_c

%include <typemaps.i>

%import "util.i"

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
#include "plothealpix.h"
#include "sip.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "log.h"
#include "fitsioutils.h"
#include "anwcs.h"
#include "coadd.h"
#include "qfits.h"
#include "mathutil.h"
#include "convolve-image.h"
#include "resample.h"
#include "cairoutils.h"
#include "an-bool.h"

#define true 1
#define false 0

%}

%apply double *OUTPUT { double *pramin, double *pramax, double *pdecmin, double *pdecmax };
%apply double *OUTPUT { double *pra, double *pdec };
%apply double *OUTPUT { double *pradius };
%apply double *OUTPUT { double *pra, double *pdec, double *pradius };
%apply double *OUTPUT { double *p_x, double *p_y };
%apply int *OUTPUT { int* p_r, int* p_g, int* p_b, int* p_a };

%include "plotstuff.h"
%include "coadd.h"
%include "qfits_image.h"
%include "fitsioutils.h"
%include "convolve-image.h"

 /*
  number* coadd_create_weight_image_from_range(const number* img, int W, int H,
  number lowval, number highval);
  */

%inline %{
PyObject* c_image_numpy_view(float* data, int nx, int ny) {
	npy_intp dims[2];
	dims[0] = ny;
	dims[1] = nx;
	return PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
  //PyArray_NewFromDescr(PyTypeObject* subtype, PyArray_Descr* descr, int nd, npy_intp* dims, npy_intp* strides, void* data, int flags, PyObject* obj)
}

%}

%pythoncode %{
def qfits_load_image(fn, ext=1, plane=0, map=1, ptype=PTYPE_FLOAT):
	ld = qfitsloader()
	ld.filename = fn
	ld.xtnum = ext
	ld.pnum = plane
	ld.map = map
	ld.ptype = ptype
	if qfitsloader_init(ld):
		raise RuntimeError('qfitsloader_init(file "%s", ext %i) failed' % (fn, ext))
	if qfits_loadpix(ld):
		raise RuntimeError('qfits_loadpix(file "%s", ext %i) failed' % (fn, ext))

	class qfits_image(object):
		def __init__(self, pix, nx, ny, ld):
			self.pix = pix
			self.nx = nx
			self.ny = ny
			self.ld = ld
		def __del__(self):
			qfitsloader_free_buffer(self.ld)
			
	return qfits_image(ld.fbuf, ld.lx, ld.ly, ld)
	%}


void free(void* ptr);

%apply int* OUTPUT { int* newW, int* newH };
#define Const
#define InlineDeclare
%include "mathutil.h"
#undef Const
#undef InlineDeclare

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
%include "plothealpix.h"
%include "sip.h"
%include "sip_qfits.h"
%include "sip-utils.h"
%include "anwcs.h"

%init %{
   import_array();
%}

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

%{
sip_t* new_sip_t(double crpix1, double crpix2, double crval1, double crval2,
            double cd11, double cd12, double cd21, double cd22) {
    sip_t* sip = sip_create();
    tan_t* tan = &(sip->wcstan);
    tan->crpix[0] = crpix1;
    tan->crpix[1] = crpix2;
    tan->crval[0] = crval1;
    tan->crval[1] = crval2;
    tan->cd[0][0] = cd11;
    tan->cd[0][1] = cd12;
    tan->cd[1][0] = cd21;
    tan->cd[1][1] = cd22;
    return sip;
}
%}

%extend sip_t {
    sip_t(double, double, double, double, double, double, double, double);

	double crval1() {
		return self->wcstan.crval[0];
	}
	double crval2() {
		return self->wcstan.crval[1];
	}

	double crpix1() {
		return self->wcstan.crpix[0];
	}
	double crpix2() {
		return self->wcstan.crpix[1];
	}

	double cd11() {
		return self->wcstan.cd[0][0];
	}
	double cd12() {
		return self->wcstan.cd[0][1];
	}
	double cd21() {
		return self->wcstan.cd[1][0];
	}
	double cd22() {
		return self->wcstan.cd[1][1];
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

	void image_add(float* img, int W, int H, float val) {
		int i;
		for (i=0; i<(W*H); i++)
			img[i] += val;
	}

	void image_weighted_smooth(float* img, int W, int H, const float* weight,
							   float sigma) {
		int K0, NK;
		float* kernel = convolve_get_gaussian_kernel_f(sigma, 5., &K0, &NK);
		convolve_separable_weighted_f(img, W, H, weight, kernel, K0, NK, img, NULL);
		free(kernel);
	}


	%}


%extend plot_args {
	PyObject* get_image_as_numpy() {
		npy_intp dim[3];
		unsigned char* img;
		PyObject* npimg;
		dim[0] = self->H;
		dim[1] = self->W;
		dim[2] = 4;
		img = cairo_image_surface_get_data(self->target);
		npimg = PyArray_EMPTY(3, dim, NPY_UBYTE, 0);
		cairoutils_argb32_to_rgba_2(img, PyArray_DATA(npimg), self->W, self->H);
		return npimg;
	}

	int set_wcs_file(const char* fn, int ext) {
		return plotstuff_set_wcs_file(self, fn, ext);
	}

    int set_size_from_wcs() {
      return plotstuff_set_size_wcs(self);
    }

	void loginit(int level) {
		log_init(level);
	}

}

%extend annotation_args {
	void add_target(double ra, double dec, const char* name) {
		plot_annotations_add_target(self, ra, dec, name);
	}
}

%extend plotoutline_args {
	int set_wcs_file(const char* fn, int ext) {
		return plot_outline_set_wcs_file(self, fn, ext);
	}
	int set_wcs_size(int W, int H) {
		return plot_outline_set_wcs_size(self, W, H);
	}
}
%pythoncode %{
def plotoutline_setattr(self, name, val):
	#print 'plotoutline_setattr', name, '=', val
	if name == 'wcs_file':
		if type(val) is tuple:
			(fn,ext) = val
		else:
			fn = val
			ext = 0
		#print 'setting outline wcs file %s, ext %i' % (fn, ext)
		plot_outline_set_wcs_file(self, fn, ext)
		return
	self.__swig__setattr__(name, val)

plotoutline_args.__swig__setattr__ = plotoutline_args.__setattr__
plotoutline_args.__setattr__ = plotoutline_setattr

	%}



%extend plotimage_args {
	int _set_image_from_numpy(PyObject* arr) {
		// Pirate array
		PyObject* yarr;
		int hasalpha = 0;
		int i, N;
		unsigned char* src;

		// MAGIC 3: min_depth and max_depth (number of dims)
		yarr = PyArray_FROMANY(arr, NPY_UBYTE, 3, 3,
							   NPY_C_CONTIGUOUS | NPY_ALIGNED);
		if (!yarr) {
			PyErr_SetString(PyExc_ValueError, "Array must be 3-dimensional ubyte");
			return -1;
		}

		switch (PyArray_DIM(yarr, 2)) {
			// RGB
		case 3:
			hasalpha = 0;
			break;
			// RGBA
		case 4:
			hasalpha = 1;
			break;
		default:
			PyErr_SetString(PyExc_ValueError, "Array must be RGB or RGBA");
			return -1;
		}
		src = PyArray_DATA(yarr);
		if (self->img) {
			free(self->img);
		}
		self->H = PyArray_DIM(yarr, 0);
		self->W = PyArray_DIM(yarr, 1);
		//printf("Allocating new %i x %i image\n", self->W, self->H);
		self->img = malloc(self->W * self->H * 4);
		N = self->W * self->H;
		for (i=0; i<N; i++) {
			if (hasalpha)
				memcpy(self->img + 4*i, src + 4*i, 4);
			else {
				memcpy(self->img + 4*i, src + 3*i, 3);
				self->img[4*i+3] = 255;
			}
		}
		Py_DECREF(yarr);
		return 0;
	}

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

%pythoncode %{
def plotimage_set_image_from_numpy(self, img):
    rtn = self._set_image_from_numpy(img)
    if rtn:
        raise RuntimeError('set_image_from_numpy() failed')

plotimage_args.set_image_from_numpy = plotimage_set_image_from_numpy
%}


%extend plotindex_args {
 int add_file(const char* fn) {
  return plot_index_add_file(self, fn);
}
}
