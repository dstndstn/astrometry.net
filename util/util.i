
%module(package="astrometry.util") util

%include <typemaps.i>
%include <cstring.i>

%{
// numpy.
#include <numpy/arrayobject.h>
#include <stdint.h>

#include "log.h"
#include "healpix.h"
#include "healpix-utils.h"
#include "anwcs.h"
#include "sip.h"
#include "fitsioutils.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "index.h"
#include "quadfile.h"
#include "codekd.h"
#include "starkd.h"
#include "starutil.h"
#include "an-bool.h"

#include "coadd.h"
#include "wcs-resample.h"
#include "resample.h"

#define true 1
#define false 0

// For sip.h
static void checkorder(int i, int j) {
	assert(i >= 0);
	assert(i < SIP_MAXORDER);
	assert(j >= 0);
	assert(j < SIP_MAXORDER);
}


%}

%init %{
	  // numpy
	  import_array();
%}

// Things in keywords.h (used by healpix.h)
#define Const
#define WarnUnusedResult
#define InlineDeclare
#define Flatten
#define ASTROMETRY_KEYWORDS_H
#define ATTRIB_FORMAT(x,y,z)

void log_init(int level);
int log_get_level();
void log_set_level(int lvl);

%include "coadd.h"
%include "resample.h"
%include "an-bool.h"

%inline %{
#define ERR(x, ...)								\
	printf(x, ## __VA_ARGS__)

	static void print_array(PyObject* arr) {
		PyArrayObject *obj;
		int i;
		PyArray_Descr *desc;
	    printf("Array: %p\n", arr);
		if (!arr) return;
		if (!PyArray_Check(arr)) {
		    printf("  Not a Numpy Array\n");
			if (arr == Py_None)
				printf("  is None\n");
			return;
		}
		printf("  Contiguous: %s\n",
			   PyArray_ISCONTIGUOUS(arr) ? "yes" : "no");
		printf("  Writeable: %s\n",
			   PyArray_ISWRITEABLE(arr) ? "yes" : "no");
		printf("  Aligned: %s\n",
			   PyArray_ISALIGNED(arr) ? "yes" : "no");
		printf("  C array: %s\n",
			   PyArray_ISCARRAY(arr) ? "yes" : "no");

		//printf("  typeobj: %p (float is %p)\n", arr->typeobj,
		//&PyFloat_Type);

		obj = (PyArrayObject*)arr;

		printf("  data: %p\n", obj->data);
		printf("  N dims: %i\n", obj->nd);
		for (i=0; i<obj->nd; i++)
			printf("  dim %i: %i\n", i, (int)obj->dimensions[i]);
		for (i=0; i<obj->nd; i++)
			printf("  stride %i: %i\n", i, (int)obj->strides[i]);
		desc = obj->descr;
		printf("  descr kind: '%c'\n", desc->kind);
		printf("  descr type: '%c'\n", desc->type);
		printf("  descr byteorder: '%c'\n", desc->byteorder);
		printf("  descr elsize: %i\n", desc->elsize);
	}

	static int lanczos_shift_image_c(PyObject* np_img, PyObject* np_weight,
									 PyObject* np_outimg,
									 PyObject* np_outweight,
									 int order, double dx, double dy) {
		int W,H;
		int i,j;

		lanczos_args_t lanczos;

		PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
		// in numpy v2.0 these constants have a NPY_ARRAY_ prefix
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED |
			   NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
		double *img, *weight, *outimg, *outweight;
		weight = NULL;
        outweight = NULL;
		lanczos.order = order;

		/*
		 printf("np_img:\n");
		 print_array(np_img);
		 printf("np_weight:\n");
		 print_array(np_weight);
		 printf("np_outimg:\n");
		 print_array(np_outimg);
		 printf("np_outweight:\n");
		 print_array(np_outweight);
		 */

		// FIXME ???? do the CheckFromAny() calls need INCREFS on the dtypes?

		np_img = PyArray_CheckFromAny(np_img, dtype, 2, 2, req, NULL);
		if (np_weight != Py_None) {
			np_weight = PyArray_CheckFromAny(np_weight, dtype, 2, 2, req, NULL);
			if (!np_weight) {
				ERR("Failed to run PyArray_FromAny on np_weight\n");
				return -1;
			}
		}
		np_outimg = PyArray_CheckFromAny(np_outimg, dtype, 2, 2, reqout, NULL);
		if (np_outweight != Py_None) {
			np_outweight = PyArray_CheckFromAny(np_outweight, dtype, 2, 2, reqout, NULL);
		}

		if (!np_img || !np_outimg || !np_outweight) {
			ERR("Failed to PyArray_FromAny the images (np_img=%p, np_outimg=%p, np_outweight=%p)\n",
				np_img, np_outimg, np_outweight);
			return -1;
		}

		H = (int)PyArray_DIM(np_img, 0);
		W = (int)PyArray_DIM(np_img, 1);

		if ((PyArray_DIM(np_outimg, 0) != H) ||
			(PyArray_DIM(np_outimg, 1) != W)) {
			ERR("All images must have the same dimensions.\n");
			return -1;
		}
		if (np_weight != Py_None) {
			if ((PyArray_DIM(np_weight, 0) != H) ||
				(PyArray_DIM(np_weight, 1) != W)) {
				ERR("All images must have the same dimensions.\n");
				return -1;
			}
			weight    = PyArray_DATA(np_weight);
		}
		if (np_outweight != Py_None) {
			if ((PyArray_DIM(np_outweight, 0) != H) ||
				(PyArray_DIM(np_outweight, 1) != W)) {
				ERR("All images must have the same dimensions.\n");
				return -1;
			}
			outweight = PyArray_DATA(np_outweight);
	    }

		/*
		 printf("np_img:\n");
		 print_array(np_img);
		 printf("np_weight:\n");
		 print_array(np_weight);
		 printf("np_outimg:\n");
		 print_array(np_outimg);
		 printf("np_outweight:\n");
		 print_array(np_outweight);
		 printf("weight = %p, outweight = %p\n", weight, outweight);
		 */

		img       = PyArray_DATA(np_img);
		outimg    = PyArray_DATA(np_outimg);

		for (i=0; i<H; i++) {
			for (j=0; j<W; j++) {
				double wt, val;
				double px, py;
				px = j - dx;
				py = i - dy;
				val = lanczos_resample_d(px, py, img, weight, W, H, &wt,
										 &lanczos);
				//printf("pixel %i,%i: wt %g\n", j, i, wt);
				if (outweight) {
				    outimg[i*W + j] = val;
				    outweight[i*W + j] = wt;
				} else {
				    outimg[i*W + j] = val / wt;
				}
			}
		}

		/*
		 if (np_img != Py_None) {
		 Py_XDECREF(np_img);
		 }
		 if (np_weight != Py_None) {
		 Py_XDECREF(np_weight);
		 }
		 if (np_outweight != Py_None) {
		 Py_XDECREF(np_outweight);
		 }
		 if (np_outimg != Py_None) {
		 Py_XDECREF(np_outimg);
		 }
		 */
		return 0;
	}
	%}

%pythoncode %{

def lanczos_shift_image(img, dx, dy, order=3, weight=None,
						outimg=None, outweight=None):
    img = img.astype(float)
    if weight is not None:
        weight = weight.astype(float)
        assert(img.shape == weight.shape)
    if outimg is None:
        outimg = np.zeros_like(img)
    if outweight is not None:
		assert(outweight.shape == img.shape)

	# print 'outweight:', outweight

    lanczos_shift_image_c(img, weight, outimg, outweight, order, dx, dy)
    if outweight is None:
        return outimg
    return outimg,outweight
	%}

// for quadfile_get_stars(quadfile* qf, int quadid, unsigned int* stars)
// --> list of stars
// swap the int* neighbours arg for tempneigh
%typemap(in, numinputs=0) unsigned int *stars (unsigned int tempstars[DQMAX]) {
	$1 = tempstars;
}
// in the argout typemap we don't know about the swap (but that's ok)
%typemap(argout) (const quadfile* qf, unsigned int quadid, unsigned int *stars) {
  int i;
  int D;
  if (result == -1) {
	  goto fail;
  }
  D = $1->dimquads;
  $result = PyList_New(D);
  for (i = 0; i < D; i++) {
	  PyObject *o = PyInt_FromLong($3[i]);
	  PyList_SetItem($result, i, o);
  }
}


/**
 double* startree_get_data_column(startree_t* s, const char* colname, const int* indices, int N);
 -> list of doubles.
 -> ASSUME indices = None
 */
%typemap(argout) (startree_t* s, const char* colname, const int* indices, int N) {
	int i;
	int N;
	if (!result) {
		goto fail;
	}
	N = $4;
	$result = PyList_New(N);
	for (i = 0; i < N; i++) {
		PyObject *o = PyFloat_FromDouble(result[i]);
		PyList_SetItem($result, i, o);
	}
	free(result);
}


%include "index.h"
%include "quadfile.h"
%include "codekd.h"
%include "starkd.h"
 //%include "qidxfile.h"

%apply double *OUTPUT { double *dx, double *dy };
%apply double *OUTPUT { double *ra, double *dec };

// for int healpix_get_neighbours(int hp, int* neigh, int nside)
// --> list of neigh
// swap the int* neighbours arg for tempneigh
%typemap(in, numinputs=0) int *neighbours (int tempneigh[8]) {
	$1 = tempneigh;
}
// in the argout typemap we don't know about the swap (but that's ok)
%typemap(argout) int *neighbours {
  int i;
  int nn;
  // convert $result to nn
  //nn = (int)PyInt_AsLong($result);
  nn = result;
  $result = PyList_New(nn);
  for (i = 0; i < nn; i++) {
	  PyObject *o = PyInt_FromLong($1[i]);
	  PyList_SetItem($result, i, o);
  }
}


// for il* healpix_rangesearch_radec(ra, dec, double, int nside, il* hps);
// --> list
// swallow the int* hps arg
%typemap(in, numinputs=0) il* hps {
	$1 = NULL;
}
%typemap(out) il* {
  int i;
  int N;
  N = il_size($1);
  $result = PyList_New(N);
  for (i = 0; i < N; i++) {
	  PyObject *o = PyInt_FromLong(il_get($1, i));
	  PyList_SetItem($result, i, o);
  }
}

%include "healpix.h"
%include "healpix-utils.h"


// anwcs_get_radec_center_and_radius
%apply double *OUTPUT { double *p_ra, double *p_dec, double *p_radius };

// anwcs_get_radec_bounds
%apply double *OUTPUT { double* pramin, double* pramax, double* pdecmin, double* pdecmax };

%apply double *OUTPUT { double *p_x, double *p_y, double *p_z };
%apply double *OUTPUT { double *p_ra, double *p_dec };
//%apply double *OUTPUT { double *xyz };

// eg anwcs_radec2pixelxy
%apply double *OUTPUT { double *p_x, double *p_y };

// anwcs_pixelxy2xyz
%typemap(in, numinputs=0) double* p_xyz (double tempxyz[3]) {
	$1 = tempxyz;
}
// in the argout typemap we don't know about the swap (but that's ok)
%typemap(argout) double* p_xyz {
  $result = Py_BuildValue("(ddd)", $1[0], $1[1], $1[2]);
}

%typemap(in, numinputs=0) char **stringparam (char* tempstr) {
			 $1 = &tempstr;
}
%typemap(in, numinputs=0) int *stringsizeparam (int slen) {
			 $1 = &slen;
}
char* anwcs_wcslib_to_string(const anwcs_t* wcs,
	  char **stringparam, int *stringsizeparam);

%ignore anwcs_wcslib_to_string;
%include "anwcs.h"

%extend anwcs_t {
	anwcs_t(char* fn, int ext=0, int slen=0) {
		if (ext == -1) {
			# assume header string
			return anwcs_wcslib_from_string(fn, slen);
		}
		anwcs_t* w = anwcs_open(fn, ext);
		return w;
	}
	~anwcs_t() { free($self); }

	void get_center(double *p_ra, double *p_dec) {
		anwcs_get_radec_center_and_radius($self, p_ra, p_dec, NULL);
	}

	anbool is_inside(double ra, double dec) {
		return anwcs_radec_is_inside_image($self, ra, dec);
	}
	double get_width() {
		return anwcs_imagew($self);
	}
	double get_height() {
		return anwcs_imageh($self);
	}
	void set_width(int W) {
		int H = anwcs_imageh($self);
		anwcs_set_size($self, W, H);
	}
	void set_height(int H) {
		int W = anwcs_imagew($self);
		anwcs_set_size($self, W, H);
	}
	void pixelxy2radec(double x, double y, double *p_ra, double *p_dec) {
		anwcs_pixelxy2radec($self, x, y, p_ra, p_dec);
	}
	int radec2pixelxy(double ra, double dec, double *p_x, double *p_y) {
		return anwcs_radec2pixelxy($self, ra, dec, p_x, p_y);
	}

 }
%pythoncode %{
anwcs = anwcs_t
anwcs.imagew = property(anwcs.get_width,  anwcs.set_width,  None, 'image width')
anwcs.imageh = property(anwcs.get_height, anwcs.set_height, None, 'image height')

def anwcs_from_string(s):
    return anwcs_t(s, -1, len(s))

def anwcs_get_header_string(self):
	s = anwcs_wcslib_to_string(self)
	return (s +
		 'NAXIS   = 2' + ' '*69 +
		 'NAXIS1  = % 20i' % self.imagew + ' '*50 +
		 'NAXIS2  = % 20i' % self.imageh + ' '*50 +
		 'END'+' '*77)
anwcs.getHeaderString = anwcs_get_header_string


	%}



%include "starutil.h"

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

%extend sip_t {
	sip_t(const char* fn=NULL, int ext=0) {
		if (fn)
			return sip_read_header_file_ext(fn, ext, NULL);
		sip_t* t = (sip_t*)calloc(1, sizeof(sip_t));
		return t;
	}

	// from string -- third arg is just to distinguish this signature.
	sip_t(const char* s, int len, int XXX) {
		return sip_from_string(s, len, NULL);
	}

	// copy constructor
	sip_t(const sip_t* other) {
		sip_t* t = (sip_t*)calloc(1, sizeof(sip_t));
		memcpy(t, other, sizeof(sip_t));
		return t;
	}

	sip_t(const tan_t* other) {
		sip_t* t = (sip_t*)calloc(1, sizeof(sip_t));
		memcpy(&(t->wcstan), other, sizeof(tan_t));
		return t;
	}

	~sip_t() { free($self); }

	double pixel_scale() { return sip_pixel_scale($self); }

	int write_to(const char* filename) {
		return sip_write_to_file($self, filename);
	}

	int ensure_inverse_polynomials() {
		return sip_ensure_inverse_polynomials($self);
	}

	/*
	 double* get_cd_matrix() {
	 return $self->wcstan.cd;
	 }
	 */

	void pixelxy2xyz(double x, double y, double *p_x, double *p_y, double *p_z) {
		double xyz[3];
		sip_pixelxy2xyzarr($self, x, y, xyz);
		*p_x = xyz[0];
		*p_y = xyz[1];
		*p_z = xyz[2];
	}
	void pixelxy2radec(double x, double y, double *p_ra, double *p_dec) {
		sip_pixelxy2radec($self, x, y, p_ra, p_dec);
	}
	int radec2pixelxy(double ra, double dec, double *p_x, double *p_y) {
		return sip_radec2pixelxy($self, ra, dec, p_x, p_y);
	}
	int radec2iwc(double ra, double dec, double *p_x, double *p_y) {
		return sip_radec2iwc($self, ra, dec, p_x, p_y);
	}
	int xyz2pixelxy(double x, double y, double z, double *p_x, double *p_y) {
		double xyz[3];
		xyz[0] = x;
		xyz[1] = y;
		xyz[2] = z;
		return sip_xyzarr2pixelxy($self, xyz, p_x, p_y);
	}

	void set_a_term(int i, int j, double val) {
		checkorder(i, j);
		$self->a[i][j] = val;
	}
	void set_b_term(int i, int j, double val) {
		checkorder(i, j);
		$self->b[i][j] = val;
	}
	void set_ap_term(int i, int j, double val) {
		checkorder(i, j);
		$self->ap[i][j] = val;
	}
	void set_bp_term(int i, int j, double val) {
		checkorder(i, j);
		$self->bp[i][j] = val;
	}

	double get_a_term(int i, int j) {
		checkorder(i, j);
		return $self->a[i][j];
	}
	double get_b_term(int i, int j) {
		checkorder(i, j);
		return $self->b[i][j];
	}
	double get_ap_term(int i, int j) {
		checkorder(i, j);
		return $self->ap[i][j];
	}
	double get_bp_term(int i, int j) {
		checkorder(i, j);
		return $self->bp[i][j];
	}

	double get_width() {
		return $self->wcstan.imagew;
	}
	double get_height() {
		return $self->wcstan.imageh;
	}
	void get_distortion(double x, double y, double *p_x, double *p_y) {
		return sip_pixel_distortion($self, x, y, p_x, p_y);
	}

	int write_to(const char* filename) {
		return sip_write_to_file($self, filename);
	}


 }
%pythoncode %{

def sip_t_tostring(self):
	tan = self.wcstan
	ct = 'SIN' if tan.sin else 'TAN'
	return (('SIP(%s): crpix (%.1f, %.1f), crval (%g, %g), cd (%g, %g, %g, %g), '
			 + 'image %g x %g; SIP orders A=%i, B=%i, AP=%i, BP=%i') %
			(ct, tan.crpix[0], tan.crpix[1], tan.crval[0], tan.crval[1],
			 tan.cd[0], tan.cd[1], tan.cd[2], tan.cd[3],
			 tan.imagew, tan.imageh, self.a_order, self.b_order,
			 self.ap_order, self.bp_order))
sip_t.__str__ = sip_t_tostring

def sip_t_get_cd(self):
    cd = self.wcstan.cd
    return (cd[0], cd[1], cd[2], cd[3])
sip_t.get_cd = sip_t_get_cd

def sip_t_radec_bounds(self):
	W,H = self.wcstan.imagew, self.wcstan.imageh
	r,d = self.pixelxy2radec([1, W, W, 1], [1, 1, H, H])
	return (r.min(), r.max(), d.min(), d.max())
sip_t.radec_bounds = sip_t_radec_bounds	   

#def sip_t_fromstring(s):
#	sip = sip_from_string(s, len(s),

Sip = sip_t
	%}

%extend tan_t {
	tan_t(char* fn=NULL, int ext=0, int only=0) {
		if (fn) {
			if (only) {
				return tan_read_header_file_ext_only(fn, ext, NULL);
			} else {
				return tan_read_header_file_ext(fn, ext, NULL);
			}
		}
		tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
		return t;
	}
	tan_t(double crval1, double crval2, double crpix1, double crpix2,
		  double cd11, double cd12, double cd21, double cd22,
		  double imagew, double imageh) {
		tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
		t->crval[0] = crval1;
		t->crval[1] = crval2;
		t->crpix[0] = crpix1;
		t->crpix[1] = crpix2;
		t->cd[0][0] = cd11;
		t->cd[0][1] = cd12;
		t->cd[1][0] = cd21;
		t->cd[1][1] = cd22;
		t->imagew = imagew;
		t->imageh = imageh;
		return t;
	}
	tan_t(const tan_t* other) {
		tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
		memcpy(t, other, sizeof(tan_t));
		return t;
	}

	~tan_t() { free($self); }
	void set(double crval1, double crval2,
		  double crpix1, double crpix2,
		  double cd11, double cd12, double cd21, double cd22,
		  double imagew, double imageh) {
		$self->crval[0] = crval1;
		$self->crval[1] = crval2;
		$self->crpix[0] = crpix1;
		$self->crpix[1] = crpix2;
		$self->cd[0][0] = cd11;
		$self->cd[0][1] = cd12;
		$self->cd[1][0] = cd21;
		$self->cd[1][1] = cd22;
		$self->imagew = imagew;
		$self->imageh = imageh;
	}

	anbool is_inside(double ra, double dec) {
		return tan_is_inside_image($self, ra, dec);
	}

	tan_t* scale(double factor) {
		tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
		tan_scale($self, t, factor);
		return t;
	}
	double get_width() {
		return $self->imagew;
	}
	double get_height() {
		return $self->imageh;
	}
	double pixel_scale() { return tan_pixel_scale($self); }
	void radec_center(double *p_ra, double *p_dec) {
		tan_get_radec_center($self, p_ra, p_dec);
	}
	double radius() {
		return tan_get_radius_deg($self);
	}
	void xyzcenter(double *p_x, double *p_y, double *p_z) {
		double xyz[3];
		tan_pixelxy2xyzarr($self, 0.5+$self->imagew/2.0, 0.5+$self->imageh/2.0, xyz);
		*p_x = xyz[0];
		*p_y = xyz[1];
		*p_z = xyz[2];
	}
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
	int radec2iwc(double ra, double dec, double *p_x, double *p_y) {
		return tan_radec2iwc($self, ra, dec, p_x, p_y);
	}
	int xyz2pixelxy(double x, double y, double z, double *p_x, double *p_y) {
		double xyz[3];
		xyz[0] = x;
		xyz[1] = y;
		xyz[2] = z;
		return tan_xyzarr2pixelxy($self, xyz, p_x, p_y);
	}
	int write_to(const char* filename) {
		return tan_write_to_file($self, filename);
	}
	void set_crval(double ra, double dec) {
		$self->crval[0] = ra;
		$self->crval[1] = dec;
	}
	void set_crpix(double x, double y) {
		$self->crpix[0] = x;
		$self->crpix[1] = y;
	}
	void set_cd(double cd11, double cd12, double cd21, double cd22) {
		$self->cd[0][0] = cd11;
		$self->cd[0][1] = cd12;
		$self->cd[1][0] = cd21;
		$self->cd[1][1] = cd22;
	}
	void set_imagesize(double w, double h) {
		$self->imagew = w;
		$self->imageh = h;
	}

	/*
	 double* get_cd_matrix() {
	 return $self->cd;
	 }
	 */


 };


%inline %{

	static int tan_wcs_resample(tan_t* inwcs, tan_t* outwcs,
								PyObject* np_inimg, PyObject* np_outimg,
								int weighted, int lorder) {
		PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
		// in numpy v2.0 these constants have a NPY_ARRAY_ prefix
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;

		Py_INCREF(dtype);
		Py_INCREF(dtype);
		np_inimg = PyArray_CheckFromAny(np_inimg, dtype, 2, 2, req, NULL);
		np_outimg = PyArray_CheckFromAny(np_outimg, dtype, 2, 2, reqout, NULL);
		if (!np_inimg || !np_outimg) {
			ERR("Failed to PyArray_FromAny the images (np_inimg=%p, np_outimg=%p)\n",
				np_inimg, np_outimg);
			Py_XDECREF(np_inimg);
			Py_XDECREF(np_outimg);
			Py_DECREF(dtype);
			return -1;
		}

		int inW, inH, outW, outH;
		float *inimg, *outimg;
		inH = (int)PyArray_DIM(np_inimg, 0);
		inW = (int)PyArray_DIM(np_inimg, 1);
		outH = (int)PyArray_DIM(np_outimg, 0);
		outW = (int)PyArray_DIM(np_outimg, 1);
		inimg = PyArray_DATA(np_inimg);
		outimg = PyArray_DATA(np_outimg);

		anwcs_t* inanwcs = anwcs_new_tan(inwcs);
		anwcs_t* outanwcs = anwcs_new_tan(outwcs);

		int res = resample_wcs(inanwcs, inimg, inW, inH,
							   outanwcs, outimg, outW, outH,
							   weighted, lorder);

		anwcs_free(inanwcs);
		anwcs_free(outanwcs);

		Py_DECREF(dtype);
		Py_DECREF(np_inimg);
		Py_DECREF(np_outimg);

		return res;
	}


	static int tansip_numpy_pixelxy2radec(tan_t* tan, sip_t* sip, PyObject* npx, PyObject* npy, 
										  PyObject* npra, PyObject* npdec, int reverse, int iwc) {
										
		int i, N;
		double *x, *y, *ra, *dec;
		assert(tan || sip);
		assert(!(tan && sip));
		
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
		if (!reverse) {
			if (iwc) {
				PyErr_SetString(PyExc_ValueError, "reverse=0, iwc=1 not supported (yet)");
				return -1;
			}
			if (tan) {
				for (i=0; i<N; i++)
					tan_pixelxy2radec(tan, x[i], y[i], ra+i, dec+i);
			} else {
				for (i=0; i<N; i++)
					sip_pixelxy2radec(sip, x[i], y[i], ra+i, dec+i);
			}
		} else {
			if (tan) {
				if (iwc) {
					for (i=0; i<N; i++)
						if (!tan_radec2iwc(tan, ra[i], dec[i], x+i, y+i)) {
							x[i] = HUGE_VAL;
							y[i] = HUGE_VAL;
						}
				} else {
					for (i=0; i<N; i++)
						if (!tan_radec2pixelxy(tan, ra[i], dec[i], x+i, y+i)) {
							x[i] = HUGE_VAL;
							y[i] = HUGE_VAL;
						}
				}
			} else {
				if (iwc) {
					for (i=0; i<N; i++)
						if (!sip_radec2iwc(sip, ra[i], dec[i], x+i, y+i)) {
							x[i] = HUGE_VAL;
							y[i] = HUGE_VAL;
						}
				} else {
					for (i=0; i<N; i++)
						if (!sip_radec2pixelxy(sip, ra[i], dec[i], x+i, y+i)) {
							x[i] = HUGE_VAL;
							y[i] = HUGE_VAL;
						}
				}
			}				
		}
		return 0;
	}

	static int tan_numpy_xyz2pixelxy(tan_t* tan, PyObject* npxyz,
		   PyObject* npx, PyObject* npy) {
		int i, N;
		int rtn = 0;
		double *x, *y;
		
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
			(PyArray_DIM(npxyz, 0) != N) ||
			(PyArray_DIM(npxyz, 1) != 3)) {
			PyErr_SetString(PyExc_ValueError, "arrays must be the same size");
			return -1;
		}
		x = PyArray_GETPTR1(npx, 0);
		y = PyArray_GETPTR1(npy, 0);
		for (i=0; i<N; i++) {
			double xyz[3];
			anbool ok;
			xyz[0] = *((double*)PyArray_GETPTR2(npxyz, i, 0));
			xyz[1] = *((double*)PyArray_GETPTR2(npxyz, i, 1));
			xyz[2] = *((double*)PyArray_GETPTR2(npxyz, i, 2));
			ok = tan_xyzarr2pixelxy(tan, xyz, x+i, y+i);
			if (!ok) {
				x[i] = -1.0;
				y[i] = -1.0;
				rtn = -1;
			}
		}
		return rtn;
	}




%}

%pythoncode %{
import numpy as np

def tan_t_tostring(self):
	ct = 'SIN' if self.sin else 'TAN'
	return ('%s: crpix (%.1f, %.1f), crval (%g, %g), cd (%g, %g, %g, %g), image %g x %g' %
			(ct, self.crpix[0], self.crpix[1], self.crval[0], self.crval[1],
			 self.cd[0], self.cd[1], self.cd[2], self.cd[3],
			 self.imagew, self.imageh))
tan_t.__str__ = tan_t_tostring

## picklable?
def tan_t_getstate(self):
	return (self.crpix[0], self.crpix[1], self.crval[0], self.crval[1],
			self.cd[0], self.cd[1], self.cd[2], self.cd[3],
			self.imagew, self.imageh)
def tan_t_setstate(self, state):
	#print 'setstate: self', self, 'state', state
	#print 'state', state
	self.this = _util.new_tan_t()
	#print 'self', repr(self)
	p0,p1,v0,v1,cd0,cd1,cd2,cd3,w,h = state
	self.set_crpix(p0,p1)
	self.set_crval(v0,v1)
	self.set_cd(cd0,cd1,cd2,cd3)
	self.set_imagesize(w,h)
	#(self.crpix[0], self.crpix[1], self.crval[0], self.crval[1],
	#self.cd[0], self.cd[1], self.cd[2], self.cd[3],
	#self.imagew, self.imageh) = state
def tan_t_getnewargs(self):
	return ()
tan_t.__getstate__ = tan_t_getstate
tan_t.__setstate__ = tan_t_setstate
tan_t.__getnewargs__ = tan_t_getnewargs

def tan_t_get_cd(self):
    cd = self.cd
    return (cd[0], cd[1], cd[2], cd[3])
tan_t.get_cd = tan_t_get_cd

def tan_t_pixelxy2radec_any(self, x, y):
	if np.iterable(x) or np.iterable(y):
		x = np.atleast_1d(x).astype(float)
		y = np.atleast_1d(y).astype(float)
		r = np.empty(len(x))
		d = np.empty(len(x))
		tansip_numpy_pixelxy2radec(self.this, None, x, y, r, d, 0, 0)
		return r,d
	else:
		return self.pixelxy2radec_single(float(x), float(y))
tan_t.pixelxy2radec_single = tan_t.pixelxy2radec
tan_t.pixelxy2radec = tan_t_pixelxy2radec_any

def tan_t_radec2pixelxy_any(self, r, d):
	if np.iterable(r) or np.iterable(d):
		r = np.atleast_1d(r).astype(float)
		d = np.atleast_1d(d).astype(float)
		# HACK -- should broadcast...
		assert(len(r) == len(d))
		x = np.empty(len(r))
		y = np.empty(len(r))
		# This looks like a bug (pixelxy2radec rather than radec2pixel)
		# but it isn't ("reverse = 1")
		tansip_numpy_pixelxy2radec(self.this, None, x, y, r, d, 1, 0)
		return x,y
	else:
		good,x,y = self.radec2pixelxy_single(r, d)
		return x,y
tan_t.radec2pixelxy_single = tan_t.radec2pixelxy
tan_t.radec2pixelxy = tan_t_radec2pixelxy_any

def tan_t_radec2iwc_any(self, r, d):
	if np.iterable(r) or np.iterable(d):
		r = np.atleast_1d(r).astype(float)
		d = np.atleast_1d(d).astype(float)
		assert(len(r) == len(d))
		x = np.empty(len(r))
		y = np.empty(len(r))
		# Call the general-purpose numpy wrapper with reverse=1, iwc=1
		tansip_numpy_pixelxy2radec(self.this, None, x, y, r, d, 1, 1)
		return x,y
	else:
		good,x,y = self.radec2iwc_single(r, d)
		return x,y
tan_t.radec2iwc_single = tan_t.radec2iwc
tan_t.radec2iwc = tan_t_radec2iwc_any






def tan_t_radec_bounds(self):
	W,H = self.imagew, self.imageh
	r,d = self.pixelxy2radec([1, W, W, 1], [1, 1, H, H])
	return (r.min(), r.max(), d.min(), d.max())
tan_t.radec_bounds = tan_t_radec_bounds	   

def tan_t_xyz2pixelxy_any(self, xyz):
	if np.iterable(xyz[0]):
		xyz = np.atleast_2d(xyz).astype(float)
		(N,three) = xyz.shape
		assert(three == 3)
		x = np.empty(N)
		y = np.empty(N)
		# This looks like a bug (pixelxy2radec rather than radec2pixel)
		# but it isn't ("reverse = 1")
		tan_numpy_xyz2pixelxy(self.this, xyz, x, y)
		return x,y
	else:
		good,x,y = self.xyz2pixelxy_single(*xyz)
		return x,y
tan_t.xyz2pixelxy_single = tan_t.xyz2pixelxy
tan_t.xyz2pixelxy = tan_t_xyz2pixelxy_any

Tan = tan_t


######## SIP #####################
def sip_t_pixelxy2radec_any(self, x, y):
	if np.iterable(x) or np.iterable(y):
		x = np.atleast_1d(x).astype(float)
		y = np.atleast_1d(y).astype(float)
		r = np.empty(len(x))
		d = np.empty(len(x))
		tansip_numpy_pixelxy2radec(None, self.this, x, y, r, d, 0, 0)
		return r,d
	else:
		return self.pixelxy2radec_single(float(x), float(y))
sip_t.pixelxy2radec_single = sip_t.pixelxy2radec
sip_t.pixelxy2radec = sip_t_pixelxy2radec_any

def sip_t_radec2pixelxy_any(self, r, d):
	if np.iterable(r) or np.iterable(d):
		r = np.atleast_1d(r).astype(float)
		d = np.atleast_1d(d).astype(float)
		# HACK -- should broadcast...
		assert(len(r) == len(d))
		x = np.empty(len(r))
		y = np.empty(len(r))
		# This looks like a bug (pixelxy2radec rather than radec2pixel)
		# but it isn't ("reverse = 1")
		tansip_numpy_pixelxy2radec(None, self.this, x, y, r, d, 1, 0)
		return x,y
	else:
		good,x,y = self.radec2pixelxy_single(r, d)
		return x,y
sip_t.radec2pixelxy_single = sip_t.radec2pixelxy
sip_t.radec2pixelxy = sip_t_radec2pixelxy_any


%} 


%include "fitsioutils.h"

