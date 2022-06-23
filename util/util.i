/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */
%module(package="astrometry.util") util

%include <typemaps.i>
%include <cstring.i>
%include <exception.i>

%{
// numpy.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "os-features.h"
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
#include "ioutils.h"

#include "coadd.h"
#include "wcs-resample.h"
#include "resample.h"
#include "keywords.h"

#include "dimage.h"

#include "fit-wcs.h"

#include "qfits_header.h"
#include "qfits_rw.h"
#include "wcs-pv2sip.h"

#define true 1
#define false 0

// For sip.h
static void checkorder(int i, int j) {
    assert(i >= 0);
    assert(i < SIP_MAXORDER);
    assert(j >= 0);
    assert(j < SIP_MAXORDER);
}

// From index.i:
/**
For returning single codes and quads as python lists, do something like this:

%typemap(out) float [ANY] {
  int i;
  $result = PyList_New($1_dim0);
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PyFloat_FromDouble((double) $1[i]);
    PyList_SetItem($result,i,o);
  }
}
**/

double* code_alloc(int DC) {
	 return malloc(DC * sizeof(double));
}
void code_free(double* code) {
	 free(code);
}
double code_get(double* code, int i) {
	return code[i];
}

long codekd_addr(index_t* ind) {
	 return (long)ind->codekd;
}
long starkd_addr(index_t* ind) {
	 return (long)ind->starkd;
}

long quadfile_addr(index_t* ind) {
	 return (long)ind->quads;
}
/*
long qidxfile_addr(qidxfile* qf) {
	 return (long)qf;
}
 */

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
%include "fit-wcs.h"

%inline %{
#define ERR(x, ...)                             \
    printf(x, ## __VA_ARGS__)

    static void print_array(PyObject* arr) {
        PyArrayObject *obj;
        int i, N;
        PyArray_Descr *desc;
        printf("Array: %p\n", arr);
        if (!arr) return;
        if (!PyArray_Check(arr)) {
            printf("  Not a Numpy Array\n");
            if (arr == Py_None)
                printf("  is None\n");
            return;
        }
        obj = (PyArrayObject*)arr;

        printf("  Contiguous: %s\n",
               PyArray_ISCONTIGUOUS(obj) ? "yes" : "no");
        printf("  Writeable: %s\n",
               PyArray_ISWRITEABLE(obj) ? "yes" : "no");
        printf("  Aligned: %s\n",
               PyArray_ISALIGNED(obj) ? "yes" : "no");
        printf("  C array: %s\n",
               PyArray_ISCARRAY(obj) ? "yes" : "no");

        //printf("  typeobj: %p (float is %p)\n", arr->typeobj,
        //&PyFloat_Type);

        printf("  data: %p\n", PyArray_DATA(obj));
        printf("  N dims: %i\n", PyArray_NDIM(obj));
        N = PyArray_NDIM(obj);
        for (i=0; i<N; i++)
            printf("  dim %i: %i\n", i, (int)PyArray_DIM(obj, i));
        for (i=0; i<N; i++)
            printf("  stride %i: %i\n", i, (int)PyArray_STRIDE(obj, i));
        desc = PyArray_DESCR(obj);
        printf("  descr kind: '%c'\n", desc->kind);
        printf("  descr type: '%c'\n", desc->type);
        printf("  descr byteorder: '%c'\n", desc->byteorder);
        printf("  descr elsize: %i\n", desc->elsize);
    }


    static PyObject* an_hist2d(PyObject* py_arrx, PyObject* py_arry,
                               PyObject* py_hist,
                               double xlo, double xhi,
                               double ylo, double yhi) {
        PyArray_Descr* dtype = NULL;
        PyArray_Descr* itype = NULL;
        int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
               NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
        int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_WRITEBACKIFCOPY;
        PyArrayObject* np_arrx;
        PyArrayObject* np_arry;
        PyArrayObject* np_hist;
        double *arrx;
        double *arry;
        int32_t *hist;
        int nx, ny;
        double dx, dy, idx, idy;
        int i, N;

        dtype = PyArray_DescrFromType(NPY_DOUBLE);
        itype = PyArray_DescrFromType(NPY_INT32);

        Py_INCREF(dtype);
        np_arrx = (PyArrayObject*)PyArray_FromAny(py_arrx, dtype, 1, 1, req, NULL);
        if (!np_arrx) {
            PyErr_SetString(PyExc_ValueError,"Expected x array to be double");
            Py_DECREF(dtype);
            return NULL;
        }
        N = PyArray_SIZE(np_arrx);

        Py_INCREF(dtype);
        np_arry = (PyArrayObject*)PyArray_FromAny(py_arry, dtype, 1, 1, req, NULL);
        if (!np_arry) {
            PyErr_SetString(PyExc_ValueError,"Expected y array to be double");
            Py_DECREF(dtype);
            Py_DECREF(np_arrx);
            return NULL;
        }
        if (PyArray_SIZE(np_arry) != N) {
            PyErr_SetString(PyExc_ValueError,"Expected x and y arrays to be the same length");
            Py_DECREF(dtype);
            Py_DECREF(np_arrx);
            return NULL;
        }
        Py_CLEAR(dtype);
        Py_INCREF(itype);
        np_hist = (PyArrayObject*)PyArray_FromAny(py_hist, itype, 2, 2, reqout, NULL);
        if (!np_hist) {
            PyErr_SetString(PyExc_ValueError,"Expected hist array to be int32");
            Py_DECREF(np_arrx);
            Py_DECREF(np_arry);
            Py_DECREF(itype);
            return NULL;
        }
        Py_CLEAR(itype);

        ny = PyArray_DIM(np_hist, 0);
        nx = PyArray_DIM(np_hist, 1);
        dx = (xhi - xlo) / (double)nx;
        dy = (yhi - ylo) / (double)ny;
        idx = 1./dx;
        idy = 1./dy;

        hist = PyArray_DATA(np_hist);
        arrx = PyArray_DATA(np_arrx);
        arry = PyArray_DATA(np_arry);

        for (i=0; i<N; i++,
                 arrx++, arry++) {
            double x, y;
            int binx, biny;
            x = *arrx;
            y = *arry;
            if ((x < xlo) || (x > xhi) || (y < ylo) || (y > yhi))
                continue;
            binx = (int)((x - xlo) * idx);
            biny = (int)((y - ylo) * idy);
            // == upper limit
            if (unlikely(binx == nx)) {
                binx--;
            }
            if (unlikely(biny == ny)) {
                biny--;
            }
            hist[biny * nx + binx]++;
        }

        Py_DECREF(np_arrx);
        Py_DECREF(np_arry);
        if (PyArray_ResolveWritebackIfCopy(np_hist) == -1) {
            PyErr_SetString(PyExc_ValueError, "Failed to write-back hist array values!");
            Py_DECREF(np_hist);
            return NULL;
        }
        Py_DECREF(np_hist);
    
        Py_RETURN_NONE;
    }


    static double flat_percentile_f(PyObject* np_arr, double pct) {
        PyArray_Descr* dtype;
        npy_intp N;
        int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
            NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
        float* x;
        float med = 0;
        int L, R;
        int mid;

        dtype = PyArray_DescrFromType(NPY_FLOAT);
        np_arr  = PyArray_CheckFromAny(np_arr, dtype, 0, 0, req, NULL);
        if (!np_arr) {
            ERR("flat_median_f: Failed to convert array to float\n");
            return 0;
        }
        dtype = NULL;
        N = PyArray_Size(np_arr);
        x = (float*)malloc(sizeof(float) * N);
        memcpy(x, PyArray_DATA((PyArrayObject*)np_arr), sizeof(float)*N);
        Py_DECREF(np_arr);

        {
            int i;
            for (i=0; i<N; i++) {
                if (!isfinite(x[i])) {
                    ERR("flat_median_f cannot handle NaN values (element %i)\n", i);
                    return x[i];
                }
            }
        }

        // Pseudocode from wikipedia's 'Selection algorithm' page
        L = 0;
        R = (int)(N-1);
        mid = (int)(pct * 0.01 * N);
        if (mid < 0) {
            mid = 0;
        }
        if (mid >= R) {
            mid = R;
        }
        while (L < R) {
            int ipivot;
            int i,j;
            int k;
            float pivot;
            //printf("L=%i, R=%i (N=%i), mid=%i\n", L, R, 1+R-L, mid);
            ipivot = random() % (1+R-L) + L;
            pivot = x[ipivot];
            // partition array...
            i = L;
            j = R;
            do {
                // scan for elements out of place
                // scan from the left:
                while (x[i] < pivot)
                    i++;
                // scan from the right:
                while (x[j] >= pivot && j>i)
                    j--;
                // now x[i] >= pivot
                // and (x[j] < pivot) OR j == i
                assert(x[i] >= pivot);
                assert((x[j] < pivot) || (j == i));
                assert(j >= i);
                if (i < j) {
                    // swap
                    float tmp = x[i];
                    x[i] = x[j];
                    x[j] = tmp;
                }
            } while (i < j);
            {
                for (k=L; k<i; k++) {
                    assert(x[k] < pivot);
                }
                for (k=i; k<=R; k++) {
                    assert(x[k] >= pivot);
                }
            }
            // partition the right partition into == and >
            j = i;
            k = R;
            do {
                // scan for elements out of place
                // scan from the right:
                while (x[k] > pivot)
                    k--;
                // scan from the left:
                while (x[j] == pivot && j<k)
                    j++;

                assert(x[k] == pivot);
                assert((x[j] > pivot) || (j == k));
                assert(k >= j);
                if (j < k) {
                    // swap
                    float tmp = x[j];
                    x[j] = x[k];
                    x[k] = tmp;
                }
            } while (j < k);

            j = k+1;

            {
                //printf("L=%i, i=%i, j=%i, k=%i, R=%i\n", L, i, j, k, R);
                for (k=L; k<i; k++) {
                    assert(x[k] < pivot);
                }
                for (k=i; k<j; k++) {
                    assert(x[k] == pivot);
                }
                for (k=j; k<=R; k++) {
                    assert(x[k] > pivot);
                }
            }



            // there must be at least one element in the right partitions
            assert(i <= R);

            // there must be at least one element in the middle partition
            assert(j-i >= 1);

            if (mid < i)
                // the median is in the left partition (< pivot)
                R = i-1;
            else if (mid >= j)
                // the median is in the right partition (> pivot)
                L = j;
            else {
                // the median is in the middle partition (== pivot)
                L = R = i;
                break;
            }
            assert(L <= mid);
            assert(R >= mid);
        }
        med = x[mid];
        free(x);
        return med;
    }

    static double flat_median_f(PyObject* np_arr) {
        return flat_percentile_f(np_arr, 50.0);
    }

    static int median_smooth(PyObject* py_image,
                             PyObject* py_mask,
                             int halfbox,
                             PyObject* py_smooth) {
        /*

         image: np.float32
         mask: bool or uint8; 1 to IGNORE.
         smooth: np.float32; output array.

         */
        PyArrayObject* np_image  = (PyArrayObject*)py_image;
        PyArrayObject* np_mask   = (PyArrayObject*)py_mask;
        PyArrayObject* np_smooth = (PyArrayObject*)py_smooth;

        if (!PyArray_Check(np_image) ||
            !PyArray_Check(np_smooth) ||
            !PyArray_ISNOTSWAPPED(np_image) ||
            !PyArray_ISNOTSWAPPED(np_smooth ) ||
            !PyArray_ISFLOAT(np_image) ||
            !PyArray_ISFLOAT(np_smooth ) ||
            (PyArray_ITEMSIZE(np_image) != sizeof(float)) ||
            (PyArray_ITEMSIZE(np_smooth ) != sizeof(float)) ||
            !(PyArray_NDIM(np_image) == 2) ||
            !(PyArray_NDIM(np_smooth ) == 2) ||
            !PyArray_ISCONTIGUOUS(np_image) ||
            !PyArray_ISCONTIGUOUS(np_smooth ) ||
            !PyArray_ISWRITEABLE(np_smooth)) {
            ERR("median_smooth: array type checks failed for image/smooth\n");
            ERR("check %i %i notswapped %i %i isfloat %i %i size %i %i ndim %i %i contig %i %i writable %i\n",
                PyArray_Check(np_image), PyArray_Check(np_smooth),
                PyArray_ISNOTSWAPPED(np_image), PyArray_ISNOTSWAPPED(np_smooth ),
                PyArray_ISFLOAT(np_image), PyArray_ISFLOAT(np_smooth),
                (PyArray_ITEMSIZE(np_image) == sizeof(float)),
                (PyArray_ITEMSIZE(np_smooth) == sizeof(float)),
                (PyArray_NDIM(np_image) == 2),
                (PyArray_NDIM(np_smooth ) == 2),
                PyArray_ISCONTIGUOUS(np_image),
                PyArray_ISCONTIGUOUS(np_smooth),
                PyArray_ISWRITEABLE(np_smooth));
            return -1;
        }
        if ((PyObject*)np_mask != Py_None) {
            if (!PyArray_Check(np_mask) ||
                !PyArray_ISNOTSWAPPED(np_mask) ||
                !PyArray_ISBOOL(np_mask) ||
                (PyArray_ITEMSIZE(np_mask) != sizeof(uint8_t)) ||
                !(PyArray_NDIM(np_mask) == 2) ||
                !PyArray_ISCONTIGUOUS(np_mask)) {
                ERR("median_smooth: array type checks failed for mask\n");
                return -1;
            }
        }
        npy_intp NX, NY;
        const float* image;
        float* smooth;
        const uint8_t* maskimg = NULL;

        NY = PyArray_DIM(np_image, 0);
        NX = PyArray_DIM(np_image, 1);
        if ((PyArray_DIM(np_smooth, 0) != NY) ||
            (PyArray_DIM(np_smooth, 1) != NX)) {
            ERR("median_smooth: 'smooth' array is wrong shape\n");
            return -1;
        }
        image = PyArray_DATA(np_image);
        smooth = PyArray_DATA(np_smooth);

        if ((PyObject*)np_mask != Py_None) {
            if ((PyArray_DIM(np_mask, 0) != NY) ||
                (PyArray_DIM(np_mask, 1) != NX)) {
                ERR("median_smooth: 'mask' array is wrong shape\n");
                return -1;
            }
            maskimg = PyArray_DATA(np_mask);
        }

        dmedsmooth(image, maskimg, (int)NX, (int)NY, halfbox, smooth);

        return 0;
    }

#define L 5

    static PyObject*lanczos5_interpolate(PyObject* np_ixi, PyObject* np_iyi,
                                         PyObject* np_dx, PyObject* np_dy,
                                         PyObject* loutputs, PyObject* linputs);
    static PyObject* lanczos5_interpolate_grid(float x0, float xstep,
                                               float y0, float ystep,
                                               PyObject* output_img,
                                               PyObject* input_img);
#include "lanczos.i"
#undef L

#define L 3

    static PyObject* lanczos3_interpolate(PyObject* np_ixi, PyObject* np_iyi,
                                          PyObject* np_dx, PyObject* np_dy,
                                          PyObject* loutputs, PyObject* linputs);
    static PyObject* lanczos3_interpolate_grid(float x0, float xstep,
                                               float y0, float ystep,
                                               PyObject* output_img,
                                               PyObject* input_img);
#include "lanczos.i"
#undef L

    static int lanczos5_filter(PyObject* py_dx, PyObject* py_f) {
        npy_intp N;
        npy_intp i;
        float* dx;
        float* f;

        PyArrayObject *np_dx = (PyArrayObject*)py_dx;
        PyArrayObject *np_f  = (PyArrayObject*)py_f;

        if (!PyArray_Check(np_dx) ||
            !PyArray_Check(np_f ) ||
            !PyArray_ISNOTSWAPPED(np_dx) ||
            !PyArray_ISNOTSWAPPED(np_f ) ||
            !PyArray_ISFLOAT(np_dx) ||
            !PyArray_ISFLOAT(np_f ) ||
            (PyArray_ITEMSIZE(np_dx) != sizeof(float)) ||
            (PyArray_ITEMSIZE(np_f ) != sizeof(float)) ||
            !(PyArray_NDIM(np_dx) == 1) ||
            !(PyArray_NDIM(np_f ) == 1) ||
            !PyArray_ISCONTIGUOUS(np_dx) ||
            !PyArray_ISCONTIGUOUS(np_f ) ||
            !PyArray_ISWRITEABLE(np_f)
            ) {
            ERR("Arrays aren't right type\n");
            return -1;
        }
        N = PyArray_DIM(np_dx, 0);
        if (PyArray_DIM(np_f, 0) != N) {
            ERR("Input and output must have same dimensions\n");
            return -1;
        }
        dx = PyArray_DATA(np_dx);
        f = PyArray_DATA(np_f);
        const double fifthpi = M_PI / 5.0;
        const double pisq = M_PI * M_PI;
        const double fiveopisq = 5. / pisq;
        for (i=N; i>0; i--, dx++, f++) {
            double x = *dx;
            if (x < -5.0 || x > 5.0) {
                *f = 0.0;
            } else if (x == 0) {
                *f = 1.0;
            } else {
                *f = fiveopisq * sin(M_PI * x) * sin(fifthpi * x) / (x * x);
            }
        }
        return 0;
    }

    static int lanczos3_filter(PyObject* py_dx, PyObject* py_f) {
        npy_intp N;
        npy_intp i;
        float* dx;
        float* f;

        PyArrayObject *np_dx = (PyArrayObject*)py_dx;
        PyArrayObject *np_f  = (PyArrayObject*)py_f;

        if (!PyArray_Check(np_dx) ||
            !PyArray_Check(np_f ) ||
            !PyArray_ISNOTSWAPPED(np_dx) ||
            !PyArray_ISNOTSWAPPED(np_f ) ||
            !PyArray_ISFLOAT(np_dx) ||
            !PyArray_ISFLOAT(np_f ) ||
            (PyArray_ITEMSIZE(np_dx) != sizeof(float)) ||
            (PyArray_ITEMSIZE(np_f ) != sizeof(float)) ||
            !(PyArray_NDIM(np_dx) == 1) ||
            !(PyArray_NDIM(np_f ) == 1) ||
            !PyArray_ISCONTIGUOUS(np_dx) ||
            !PyArray_ISCONTIGUOUS(np_f ) ||
            !PyArray_ISWRITEABLE(np_f)
            ) {
            ERR("Arrays aren't right type\n");
            return -1;
        }
        N = PyArray_DIM(np_dx, 0);
        if (PyArray_DIM(np_f, 0) != N) {
            ERR("Input and output must have same dimensions\n");
            return -1;
        }
        dx = PyArray_DATA(np_dx);
        f = PyArray_DATA(np_f);
        const double thirdpi = M_PI / 3.0;
        const double pisq = M_PI * M_PI;
        const double threeopisq = 3. / pisq;
        for (i=N; i>0; i--, dx++, f++) {
            double x = *dx;
            if (x < -3.0 || x > 3.0) {
                *f = 0.0;
            } else if (x == 0) {
                *f = 1.0;
            } else {
                *f = threeopisq * sin(M_PI * x) * sin(thirdpi * x) / (x * x);
            }
        }
        return 0;
    }

    static int lanczos3_filter_table(PyObject* py_dx, PyObject* py_f, int rangecheck) {
        npy_intp N;
        npy_intp i;
        float* dx;
        float* f;

        PyArrayObject *np_dx = (PyArrayObject*)py_dx;
        PyArrayObject *np_f  = (PyArrayObject*)py_f;

        // Nlutunit is number of bins per unit x
        static const int Nlutunit = 1024;
        static const float lut0 = -4.;
        static const int Nlut = 8192; //8 * Nlutunit;
        // We want bins to go from -4 to 4 (Lanczos-3 range of -3 to 3, plus some buffer)
        // [Nlut]
        static float lut[8192];
        static int initialized = 0;

        if (!initialized) {
            for (i=0; i<(Nlut); i++) {
                float x,f;
                x = (lut0 + (i / (float)Nlutunit));
                if (x <= -3.0 || x >= 3.0) {
                    f = 0.0;
                } else if (x == 0) {
                    f = 1.0;
                } else {
                    f = 3. * sin(M_PI * x) * sin(M_PI / 3.0 * x) / (M_PI * M_PI * x * x);
                }
                lut[i] = f;
            }
            initialized = 1;
        }

        if (!PyArray_Check(np_dx) ||
            !PyArray_Check(np_f )) {
            ERR("Array check\n");
        }
        if (!PyArray_ISNOTSWAPPED(np_dx) ||
            !PyArray_ISNOTSWAPPED(np_f )) {
            ERR("Swapped\n");
        }
        if (!PyArray_ISFLOAT(np_dx) ||
            !PyArray_ISFLOAT(np_f )) {
            ERR("Float\n");
        }
        if ((PyArray_ITEMSIZE(np_dx) != sizeof(float)) ||
            (PyArray_ITEMSIZE(np_f ) != sizeof(float))) {
            ERR("sizeof float\n");
        }
        if ((PyArray_ITEMSIZE(np_dx) != sizeof(float))) {
            ERR("sizeof dx %i\n", (int)PyArray_ITEMSIZE(np_dx));
        }
        if ((PyArray_ITEMSIZE(np_f ) != sizeof(float))) {
            ERR("sizeof f %i\n", (int)PyArray_ITEMSIZE(np_f));
        }
        if (!(PyArray_NDIM(np_dx) == 1) ||
            !(PyArray_NDIM(np_f ) == 1)) {
            ERR("one-d\n");
        }
        if (!PyArray_ISCONTIGUOUS(np_dx) ||
            !PyArray_ISCONTIGUOUS(np_f )) {
            ERR("contig\n");
        }
        if (!PyArray_ISWRITEABLE(np_f)) {
            ERR("writable\n");
        }


        if (!PyArray_Check(np_dx) ||
            !PyArray_Check(np_f ) ||
            !PyArray_ISNOTSWAPPED(np_dx) ||
            !PyArray_ISNOTSWAPPED(np_f ) ||
            !PyArray_ISFLOAT(np_dx) ||
            !PyArray_ISFLOAT(np_f ) ||
            (PyArray_ITEMSIZE(np_dx) != sizeof(float)) ||
            (PyArray_ITEMSIZE(np_f ) != sizeof(float)) ||
            !(PyArray_NDIM(np_dx) == 1) ||
            !(PyArray_NDIM(np_f ) == 1) ||
            !PyArray_ISCONTIGUOUS(np_dx) ||
            !PyArray_ISCONTIGUOUS(np_f ) ||
            !PyArray_ISWRITEABLE(np_f)
            ) {
            ERR("Arrays aren't right type\n");
            return -1;
        }
        N = PyArray_DIM(np_dx, 0);
        if (PyArray_DIM(np_f, 0) != N) {
            ERR("Input and output must have same dimensions\n");
            return -1;
        }
        dx = PyArray_DATA(np_dx);
        f = PyArray_DATA(np_f);
        if (rangecheck) {
            for (i=N; i>0; i--, dx++, f++) {
                float x = *dx;
                int li = (int)((x - lut0) * Nlutunit);
                if ((li < 0) || (li >= Nlut)) {
                    *f = 0.0;
                } else {
                    *f = lut[li];
                }
            }
        } else {
            for (i=N; i>0; i--, dx++, f++) {
                float x = *dx;
                int li = (int)((x - lut0) * Nlutunit);
                *f = lut[li];
            }
        }
        return 0;
    }

    static int lanczos_shift_image_c(PyObject* py_img,
                                     PyObject* py_weight,
                                     PyObject* py_outimg,
                                     PyObject* py_outweight,
                                     int order, double dx, double dy) {
        int W,H;
        int i,j;

        lanczos_args_t lanczos;

        PyArray_Descr* dtype;
        int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
               NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
        int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_UPDATEIFCOPY;
        double *img, *weight, *outimg, *outweight;

        PyArrayObject *np_img=NULL, *np_weight=NULL, *np_outimg=NULL, *np_outweight=NULL;

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

        dtype = PyArray_DescrFromType(NPY_DOUBLE);
        Py_INCREF(dtype);
        np_img = (PyArrayObject*)PyArray_CheckFromAny(py_img, dtype, 2, 2, req, NULL);
        if (py_weight != Py_None) {
            Py_INCREF(dtype);
            np_weight = (PyArrayObject*)PyArray_CheckFromAny(py_weight, dtype, 2, 2, req, NULL);
            if (!np_weight) {
                ERR("Failed to run PyArray_FromAny on np_weight\n");
                return -1;
            }
        }
        Py_INCREF(dtype);
        np_outimg = (PyArrayObject*)PyArray_CheckFromAny(py_outimg, dtype, 2, 2, reqout, NULL);
        if (py_outweight != Py_None) {
            Py_INCREF(dtype);
            np_outweight = (PyArrayObject*)PyArray_CheckFromAny(py_outweight, dtype, 2, 2, reqout, NULL);
        }
        Py_DECREF(dtype);
        dtype = NULL;

        if (!np_img || !np_outimg ||
            ((py_outweight != Py_None) && !np_outweight)) {
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
        if (np_weight) {
            if ((PyArray_DIM(np_weight, 0) != H) ||
                (PyArray_DIM(np_weight, 1) != W)) {
                ERR("All images must have the same dimensions.\n");
                return -1;
            }
            weight = PyArray_DATA(np_weight);
        }
        if (np_outweight) {
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
%typemap(argout) (const quadfile_t* qf, unsigned int quadid, unsigned int *stars) {
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

// for startree_get()
%typemap(in, numinputs=0) double *p_xyz (double tempxyz[3]) {
    $1 = tempxyz;
}
%apply double *OUTPUT { double *p_x, double *p_y, double *p_z };
%apply double *OUTPUT { double *p_ra, double *p_dec };
//%apply double *OUTPUT { double *xyz };

// anwcs_pixelxy2xyz, startree_get
%typemap(in, numinputs=0) double* p_xyz (double tempxyz[3]) {
    $1 = tempxyz;
}
// in the argout typemap we don't know about the swap (but that's ok)
%typemap(argout) double* p_xyz {
  $result = Py_BuildValue("(ddd)", $1[0], $1[1], $1[2]);
}

%include "index.h"
%include "quadfile.h"
%include "codekd.h"
%include "starkd.h"
 //%include "qidxfile.h"

double* code_alloc(int DC);
void code_free(double* code);
double code_get(double* code, int i);
long codekd_addr(index_t* ind);
long starkd_addr(index_t* ind);
long quadfile_addr(index_t* ind);
//long qidxfile_addr(qidxfile* qf);


%apply double *OUTPUT { double *dx, double *dy };
%apply double *OUTPUT { double *ra, double *dec };

// healpix_to_xyz
%apply double *OUTPUT { double *p_x, double *p_y, double *p_z };

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

    // healpix_radec_bounds
%apply double *OUTPUT { double *ralo, double *rahi, double *declo, double *dechi };

// xyztohealpixf
%apply double *OUTPUT { double *p_dx, double *p_dy };

%include "healpix.h"
%include "healpix-utils.h"


// anwcs_get_radec_center_and_radius
%apply double *OUTPUT { double *p_ra, double *p_dec, double *p_radius };

// anwcs_get_radec_bounds
%apply double *OUTPUT { double* pramin, double* pramax, double* pdecmin, double* pdecmax };

// eg anwcs_radec2pixelxy
%apply double *OUTPUT { double *p_x, double *p_y };

// anwcs_get_cd_matrix
%typemap(in, numinputs=0) double* p_cd (double tempcd[4]) {
    $1 = tempcd;
}
%typemap(argout) double* p_cd {
  $result = Py_BuildValue("(dddd)", $1[0], $1[1], $1[2], $1[3]);
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
        if ((ext == -1) ||
            (starts_with(fn, "SIMPLE  =") && !file_exists(fn))) {
            // assume header string
            if (slen == 0) {
                 slen = (int)strlen(fn);
            }
            return anwcs_wcslib_from_string(fn, slen);
        }
        anwcs_t* w = anwcs_open(fn, ext);
        return w;
    }
    ~anwcs_t() { free($self); }

    double pixel_scale() { return anwcs_pixel_scale($self); }

    // FIXME -- this should be more like linearizeAtPoint(x,y)
    //void get_cd() { return anwcs_get_cd_matrix($self); }

    void get_center(double *p_ra, double *p_dec) {
        anwcs_get_radec_center_and_radius($self, p_ra, p_dec, NULL);
    }
    void get_radius(double *p_radius) {
      anwcs_get_radec_center_and_radius($self, NULL, NULL, p_radius);
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

    int write_to(const char* filename) {
        return anwcs_write($self, filename);
    }

 }
%pythoncode %{
anwcs = anwcs_t
anwcs.imagew = property(anwcs.get_width,  anwcs.set_width,  None, 'image width')
anwcs.imageh = property(anwcs.get_height, anwcs.set_height, None, 'image height')
anwcs.writeto = anwcs.write_to

def anwcs_t_get_shape(self):
    return int(self.get_height()), int(self.get_width())
anwcs_t.get_shape = anwcs_t_get_shape

def anwcs_t_set_shape(self, S):
    H,W = S
    self.set_height(H)
    self.set_width(W)
anwcs_t.set_shape = anwcs_t_set_shape
anwcs_t.shape = property(anwcs_t.get_shape, anwcs_t.set_shape, None, 'image shape')

# same API as tan_t
anwcs.radec_center = anwcs.get_center
anwcs.radius = anwcs.get_radius

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

def anwcs_radec_bounds(self, stepsize=1000):
    r0,r1,d0,d1 = anwcs_get_radec_bounds(self, stepsize)
    return r0,r1,d0,d1
anwcs.radec_bounds = anwcs_radec_bounds

def anwcs_get_cd(self):
    return anwcs_get_cd_matrix(self)
anwcs.get_cd = anwcs_get_cd

    %}



%include "starutil.h"

%apply (char *STRING, int LENGTH) { (const unsigned char *, int) };

%include "qfits_header.h"
%include "qfits_rw.h"

%pythondynamic qfits_header;

%pythoncode %{
def fitsio_to_qfits_header(hdr):
    hdrstr = ''
    for rec in hdr.records():
        cardstr = rec.get('card', None)
        if cardstr is None:
            cardstr = rec.get('card_string', None)
        if cardstr is None:
            cardstr = hdr._record2card(rec)
        # pad
        cardstr = cardstr + ' '*(80 - len(cardstr))
        hdrstr += cardstr
    hdrstr += 'END' + ' '*77
    qhdr = qfits_header_read_hdr_string(hdrstr)
    return qhdr
%}

%include "wcs-pv2sip.h"

%pythoncode %{
def wcs_pv2sip_hdr(hdr, order=5, xlo=0, xhi=0, ylo=0, yhi=0,
                   stepsize=0, W=0, H=0):
    qhdr = fitsio_to_qfits_header(hdr)
    forcetan = False
    doshift = 1
    scamp = False

    sip = wcs_pv2sip_header(qhdr, None, 0, stepsize, xlo, xhi, ylo, yhi, W, H,
                            order, forcetan, doshift)
    return sip
%}


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

// SIP coefficients; array size must match SIP_MAXORDER.
%apply double flatmatrix[ANY][ANY] { double a[10][10] };
%apply double flatmatrix[ANY][ANY] { double b[10][10] };
%apply double flatmatrix[ANY][ANY] { double ap[10][10] };
%apply double flatmatrix[ANY][ANY] { double bp[10][10] };


%include "sip.h"
%include "sip_qfits.h"
%include "sip-utils.h"

%pythondynamic sip_t;

%extend sip_t {
    sip_t(const char* fn=NULL, int ext=0) {
        if (fn)
            return sip_read_header_file_ext_only(fn, ext, NULL);
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

    sip_t(const qfits_header* hdr) {
        sip_t* t = sip_read_header(hdr, NULL);
        return t;
    }

    ~sip_t() { free($self); }

    sip_t* get_subimage(int x0, int y0, int w, int h) {
        sip_t* sub = malloc(sizeof(sip_t));
        memcpy(sub, $self, sizeof(sip_t));
        sub->wcstan.crpix[0] -= x0;
        sub->wcstan.crpix[1] -= y0;
        sub->wcstan.imagew = w;
        sub->wcstan.imageh = h;
        return sub;
    }

    sip_t* scale(double factor) {
        sip_t* s = (sip_t*)calloc(1, sizeof(sip_t));
        sip_scale($self, s, factor);
        return s;
    }

    double pixel_scale() { return sip_pixel_scale($self); }

    void radec_center(double *p_ra, double *p_dec) {
        sip_get_radec_center($self, p_ra, p_dec);
    }
    double radius() {
        return sip_get_radius_deg($self);
    }

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
    void iwc2pixelxy(double u, double v, double *p_x, double *p_y) {
        sip_iwc2pixelxy($self, u, v, p_x, p_y);
    }
    void pixelxy2iwc(double x, double y, double *p_x, double *p_y) {
        sip_pixelxy2iwc($self, x, y, p_x, p_y);
    }
    void iwc2radec(double u, double v, double *p_ra, double *p_dec) {
        sip_iwc2radec($self, u, v, p_ra, p_dec);
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

    anbool is_inside(double ra, double dec) {
       return sip_is_inside_image($self, ra, dec);
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

    void set_width(double x) {
        $self->wcstan.imagew = x;
    }
    void set_height(double x) {
        $self->wcstan.imageh = x;
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
    void get_undistortion(double x, double y, double *p_x, double *p_y) {
        return sip_pixel_undistortion($self, x, y, p_x, p_y);
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

def sip_t_addtoheader(self, hdr):
    '''Adds this SIP WCS header to the given fitsio header'''
    self.wcstan.add_to_header(hdr)
    hdr.delete('CTYPE1')
    hdr.delete('CTYPE2')
    for k,v,c in [
        ('CTYPE1', 'RA---TAN-SIP', 'TANgent plane+SIP'),
        ('CTYPE2', 'DEC--TAN-SIP', 'TANgent plane+SIP'),
        ('A_ORDER', self.a_order, 'Polynomial order, axis 1'),
        ('B_ORDER', self.b_order, 'Polynomial order, axis 2'),
        ('AP_ORDER', self.ap_order, 'Inv.polynomial order, axis 1'),
        ('BP_ORDER', self.bp_order, 'Inv.polynomial order, axis 2'),
        ]:
        hdr.add_record(dict(name=k, value=v, comment=c))
    for i in range(self.a_order + 1):
        for j in range(self.a_order + 1):
            #if i + j < 1:
            # drop linear (CD) terms
            if i + j < 2:
                continue
            if i + j > self.a_order:
                continue
            hdr.add_record(dict(name='A_%i_%i' % (i,j), value=self.get_a_term(i, j),
                                comment='SIP polynomial term'))
    for i in range(self.b_order + 1):
        for j in range(self.b_order + 1):
            #if i + j < 1:
            # drop linear (CD) terms
            if i + j < 2:
                continue
            if i + j > self.b_order:
                continue
            hdr.add_record(dict(name='B_%i_%i' % (i,j), value=self.get_b_term(i, j),
                                comment='SIP polynomial term'))
    for i in range(self.ap_order + 1):
        for j in range(self.ap_order + 1):
            if i + j < 1:
                continue
            if i + j > self.ap_order:
                continue
            hdr.add_record(dict(name='AP_%i_%i' % (i,j), value=self.get_ap_term(i, j),
                                comment='SIP polynomial term'))
    for i in range(self.bp_order + 1):
        for j in range(self.bp_order + 1):
            if i + j < 1:
                continue
            if i + j > self.bp_order:
                continue
            hdr.add_record(dict(name='BP_%i_%i' % (i,j), value=self.get_bp_term(i, j),
                                comment='SIP polynomial term'))
sip_t.add_to_header = sip_t_addtoheader


# def sip_t_get_subimage(self, x0, y0, w, h):
#     wcs2 = sip_t(self)
#     cpx,cpy = wcs2.crpix
#     wcs2.set_crpix((cpx - x0, cpy - y0))
#     wcs2.set_width(float(w))
#     wcs2.set_height(float(h))
#     return wcs2
# sip_t.get_subimage = sip_t_get_subimage

def sip_t_get_shape(self):
    return (self.wcstan.imageh, self.wcstan.imagew)
sip_t.get_shape = sip_t_get_shape

def sip_t_set_shape(self, S):
    H,W = S
    self.set_height(H)
    self.set_width(W)
sip_t.set_shape = sip_t_set_shape

sip_t.imagew = property(sip_t.get_width,  sip_t.set_width,  None, 'image width')
sip_t.imageh = property(sip_t.get_height, sip_t.set_height, None, 'image height')
sip_t.shape = property(sip_t.get_shape, sip_t.set_shape, None, 'image shape')

def sip_t_get_cd(self):
    cd = self.wcstan.cd
    return (cd[0], cd[1], cd[2], cd[3])
def sip_t_set_cd(self, x):
    self.wcstan.cd = x
sip_t.get_cd = sip_t_get_cd
sip_t.set_cd = sip_t_set_cd

def sip_t_get_crval(self):
    return self.wcstan.crval
def sip_t_set_crval(self, x):
    self.wcstan.crval = x
sip_t.get_crval = sip_t_get_crval
sip_t.set_crval = sip_t_set_crval

def sip_t_get_crpix(self):
    return self.wcstan.crpix
def sip_t_set_crpix(self, x):
    self.wcstan.crpix = x
sip_t.get_crpix = sip_t_get_crpix
sip_t.set_crpix = sip_t_set_crpix

sip_t.crval = property(sip_t_get_crval, sip_t_set_crval, None, 'CRVAL')
sip_t.crpix = property(sip_t_get_crpix, sip_t_set_crpix, None, 'CRPIX')
sip_t.cd    = property(sip_t_get_cd   , sip_t_set_cd,    None, 'CD')


def sip_t_radec_bounds(self):
    # W,H = self.wcstan.imagew, self.wcstan.imageh
    # r,d = self.pixelxy2radec([1, W, W, 1], [1, 1, H, H])
    # return (r.min(), r.max(), d.min(), d.max())
    W,H = self.imagew, self.imageh
    r,d = self.pixelxy2radec([1, W/2, W, W, W, W/2, 1, 1], [1, 1, 1, H/2, H, H, H, H/2])
    rx = r.max()
    rn = r.min()
    # ugh, RA wrap-around.  We find the largest value < 180 (ie, near zero) and smallest value > 180 (ie, near 360)
    # and report them with ralo > rahi so that this case can be identified
    if rx - rn > 180:
        rx = r[r < 180].max()
        rn = r[r > 180].min()
    return (rn, rx, d.min(), d.max())
sip_t.radec_bounds = sip_t_radec_bounds

#def sip_t_fromstring(s):
#   sip = sip_from_string(s, len(s),

_real_sip_t_init = sip_t.__init__
def my_sip_t_init(self, *args, **kwargs):
    # fitsio header: check for '.records()' function.
    if len(args) == 1 and hasattr(args[0], 'records'):
        try:
            hdr = args[0]
            qhdr = fitsio_to_qfits_header(hdr)
            args = [qhdr]
        except:
            pass

    _real_sip_t_init(self, *args, **kwargs)
    if self.this is None:
        raise RuntimeError('Duck punch!')
sip_t.__init__ = my_sip_t_init


Sip = sip_t
    %}

%pythondynamic tan_t;

%extend tan_t {
    tan_t(char* fn=NULL, int ext=0, int only=0) {
        tan_t* t = NULL;
        if (fn) {
            if (only) {
                t = tan_read_header_file_ext_only(fn, ext, NULL);
            } else {
                t = tan_read_header_file_ext(fn, ext, NULL);
            }
        } else {
            t = (tan_t*)calloc(1, sizeof(tan_t));
        }
    //      printf("tan_t: %p\n", t);
        if (!t) {
            // SWIG_exception(SWIG_RuntimeError, "Failed to read TAN WCS header");
            PyErr_SetString(PyExc_RuntimeError, "Failed to read TAN WCS header");
            return NULL;
        }
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

    tan_t(const qfits_header* hdr) {
        tan_t* t = tan_read_header(hdr, NULL);
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
    tan_t* rotate(double angle_deg) {
        tan_t* t = (tan_t*)calloc(1, sizeof(tan_t));
        tan_rotate($self, t, angle_deg);
        return t;
    }
    double get_width() {
        return $self->imagew;
    }
    double get_height() {
        return $self->imageh;
    }

    void set_width(double x) {
        $self->imagew = x;
    }
    void set_height(double x) {
        $self->imageh = x;
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
    void iwc2pixelxy(double u, double v, double *p_x, double *p_y) {
        tan_iwc2pixelxy($self, u, v, p_x, p_y);
    }
    void pixelxy2iwc(double x, double y, double *p_x, double *p_y) {
        tan_pixelxy2iwc($self, x, y, p_x, p_y);
    }
    void iwc2radec(double u, double v, double *p_ra, double *p_dec) {
        tan_iwc2radec($self, u, v, p_ra, p_dec);
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

  // Wrapper on coadd_add_image that accepts numpy arrays.

  static int coadd_add_numpy(coadd_t* c, 
                             PyObject* py_img, PyObject* py_weight,
                             float fweight, const anwcs_t* wcs) {
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
    float *img, *weight=NULL;

    PyArrayObject *np_img=NULL, *np_weight=NULL;

    Py_INCREF(dtype);
    np_img = (PyArrayObject*)PyArray_CheckFromAny(py_img, dtype, 2, 2, req, NULL);
    img = PyArray_DATA(np_img);
    if (!np_img) {
      ERR("Failed to PyArray_FromAny the image\n");
      Py_XDECREF(np_img);
      Py_DECREF(dtype);
      return -1;
    }
    if (py_weight != Py_None) {
      Py_INCREF(dtype);
      np_weight = (PyArrayObject*)PyArray_CheckFromAny(py_weight, dtype, 2, 2, req, NULL);
      if (!np_weight) {
        ERR("Failed to PyArray_FromAny the weight\n");
        Py_XDECREF(np_weight);
        Py_DECREF(dtype);
        return -1;
      }
      weight = PyArray_DATA(np_weight);
    }

    int rtn = coadd_add_image(c, img, weight, fweight, wcs);

    Py_DECREF(np_img);
    if (weight) {
      Py_DECREF(np_weight);
    }
    Py_DECREF(dtype);
    return rtn;
  }

  static PyObject* coadd_get_snapshot_numpy(coadd_t* co, float badpix) {
    npy_intp dim[2];
    PyObject* npimg;
    dim[0] = co->H;
    dim[1] = co->W;
    npimg = PyArray_EMPTY(2, dim, NPY_FLOAT, 0);
    coadd_get_snapshot(co, PyArray_DATA((PyArrayObject*)npimg), badpix);
    return npimg;
  }

  static sip_t* fit_sip_wcs_py(PyObject* py_starxyz,
                              PyObject* py_fieldxy,
                              PyObject* py_weights,
                              tan_t* tanin,
                              int sip_order,
                              int inv_order) {
      PyArray_Descr* dtype = PyArray_DescrFromType(NPY_DOUBLE);
      int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED |
          NPY_ARRAY_ELEMENTSTRIDES;
      PyArrayObject *np_starxyz=NULL, *np_fieldxy=NULL, *np_weights=NULL;

      Py_INCREF(dtype);
      np_starxyz = (PyArrayObject*)PyArray_CheckFromAny(py_starxyz, dtype, 2, 2, req, NULL);
      Py_INCREF(dtype);
      np_fieldxy = (PyArrayObject*)PyArray_CheckFromAny(py_fieldxy, dtype, 2, 2, req, NULL);
      if (!np_starxyz || !np_fieldxy) {
          Py_DECREF(dtype);
          Py_DECREF(dtype);
          printf("Failed to convert starxyz or fieldxy to numpy double arrays\n");
          return NULL;
      }
      if (py_weights != Py_None) {
          Py_INCREF(dtype);
          np_weights = (PyArrayObject*)PyArray_CheckFromAny(py_weights, dtype, 1, 1, req,NULL);
          if (!np_weights) {
              Py_DECREF(dtype);
              printf("Failed to convert weights to numpy double array\n");
              return NULL;
          }
      }
      Py_DECREF(dtype);
      dtype = NULL;

      int M = (int)PyArray_DIM(np_starxyz, 0);
      if (PyArray_DIM(np_fieldxy, 0) != M) {
          printf("Expected starxyz and fieldxy to have the same length\n");
          return NULL;
      }
      if (np_weights && (PyArray_DIM(np_weights, 0) != M)) {
          printf("Expected starxyz and weights to have the same length\n");
          return NULL;
      }
      if ((PyArray_DIM(np_starxyz, 1) != 3) ||
          (PyArray_DIM(np_fieldxy, 1) != 2)) {
          printf("Expected starxyz Mx3 and fieldxy Mx2\n");
          return NULL;
      }

      sip_t* sipout = calloc(1, sizeof(sip_t));

      double* weights = NULL;
      if (np_weights)
          weights = PyArray_DATA(np_weights);

      int doshift = 1;
      int rtn = fit_sip_wcs(PyArray_DATA(np_starxyz),
                            PyArray_DATA(np_fieldxy),
                            weights, M, tanin, sip_order, inv_order,
                            doshift, sipout);
      if (rtn) {
          free(sipout);
          printf("fit_sip_wcs() returned %i\n", rtn);
          return NULL;
      }
      return sipout;
  }


%}


%inline %{

    typedef anbool (*f_2to2ok)(const void*, double, double, double*, double*);
    typedef void   (*f_2to2)  (const void*, double, double, double*, double*);
    typedef int    (*f_2to2i) (const void*, double, double, double*, double*);

    static PyObject* broadcast_2to2ok
        (
         //anbool func(const void*, double, double, double*, double*),
         f_2to2ok func,
         const void* baton,
         PyObject* in1, PyObject* in2);

    static PyObject* broadcast_2to2
        (
         //void func(const void*, double, double, double*, double*),
         f_2to2 func,
         const void* baton,
         PyObject* in1, PyObject* in2);

    static PyObject* broadcast_2to2i
        (
         //int func(const void*, double, double, double*, double*),
         f_2to2i func,
         const void* baton,
         PyObject* in1, PyObject* in2);

static PyObject* tan_rd2xy_wrapper(const tan_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2ok((f_2to2ok)tan_radec2pixelxy, wcs, in1, in2);
}
static PyObject* sip_rd2xy_wrapper(const sip_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2ok((f_2to2ok)sip_radec2pixelxy, wcs, in1, in2);
}
static PyObject* anwcs_rd2xy_wrapper(const anwcs_t* wcs,
                                     PyObject* in1, PyObject* in2) {
    return broadcast_2to2i((f_2to2i)anwcs_radec2pixelxy, wcs, in1, in2);
}

static PyObject* tan_iwc2xy_wrapper(const tan_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)tan_iwc2pixelxy, wcs, in1, in2);
}
static PyObject* sip_iwc2xy_wrapper(const sip_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)sip_iwc2pixelxy, wcs, in1, in2);
}

static PyObject* tan_xy2iwc_wrapper(const tan_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)tan_pixelxy2iwc, wcs, in1, in2);
}
static PyObject* sip_xy2iwc_wrapper(const sip_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)sip_pixelxy2iwc, wcs, in1, in2);
}

static PyObject* tan_iwc2rd_wrapper(const tan_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)tan_iwc2radec, wcs, in1, in2);
}
static PyObject* sip_iwc2rd_wrapper(const sip_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)sip_iwc2radec, wcs, in1, in2);
}

static PyObject* tan_rd2iwc_wrapper(const tan_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2ok((f_2to2ok)tan_radec2iwc, wcs, in1, in2);
}
static PyObject* sip_rd2iwc_wrapper(const sip_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2ok((f_2to2ok)sip_radec2iwc, wcs, in1, in2);
}

static PyObject* tan_xy2rd_wrapper(const tan_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)tan_pixelxy2radec, wcs, in1, in2);
}
static PyObject* sip_xy2rd_wrapper(const sip_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2((f_2to2)sip_pixelxy2radec, wcs, in1, in2);
}
static PyObject* anwcs_xy2rd_wrapper(const anwcs_t* wcs,
                                   PyObject* in1, PyObject* in2) {
    return broadcast_2to2i((f_2to2i)anwcs_pixelxy2radec, wcs, in1, in2);
}

    static PyObject* broadcast_2to2ok
        (
         //anbool func(const void*, double, double, double*, double*),
         f_2to2ok func,
         const void* baton,
         PyObject* in1, PyObject* in2) {

        NpyIter *iter = NULL;
        NpyIter_IterNextFunc *iternext;
        PyArrayObject *op[5];
        PyObject *ret;
        npy_uint32 flags;
        npy_uint32 op_flags[5];
        npy_intp *innersizeptr;
        char **dataptrarray;
        npy_intp* strideptr;
        PyArray_Descr* dtypes[5];
        npy_intp i, N;

        // we'll do the inner loop ourselves
        flags = NPY_ITER_EXTERNAL_LOOP;
        // use buffers to satisfy dtype casts
        flags |= NPY_ITER_BUFFERED;
        // grow inner loop
        flags |= NPY_ITER_GROWINNER;

        op[0] = (PyArrayObject*)PyArray_FromAny(in1, NULL, 0, 0, 0, NULL);
        op[1] = (PyArrayObject*)PyArray_FromAny(in2, NULL, 0, 0, 0, NULL);
        // automatically allocate the output arrays.
        op[2] = NULL;
        op[3] = NULL;
        op[4] = NULL;

        if ((PyArray_Size((PyObject*)op[0]) == 0) ||
            (PyArray_Size((PyObject*)op[1]) == 0)) {
            // empty inputs -- empty outputs
            npy_intp dim = 0;
            ret = Py_BuildValue("(NNN)",
                                PyArray_SimpleNew(1, &dim, NPY_BOOL),
                                PyArray_SimpleNew(1, &dim, NPY_DOUBLE),
                                PyArray_SimpleNew(1, &dim, NPY_DOUBLE));
            goto cleanup;
        }

        op_flags[0] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_flags[1] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO;
        op_flags[3] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO;
        op_flags[4] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO;

        dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[3] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[4] = PyArray_DescrFromType(NPY_BOOL);

        iter = NpyIter_MultiNew(5, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                                op_flags, dtypes);
        for (i=0; i<5; i++)
            Py_DECREF(dtypes[i]);

        if (!iter)
            return NULL;

        iternext = NpyIter_GetIterNext(iter, NULL);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        // The inner loop size and data pointers may change during the
        // loop, so just cache the addresses.
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        dataptrarray = NpyIter_GetDataPtrArray(iter);

        do {
            // are the inputs contiguous?  (Outputs will be, since we
            // allocated them)
            if ((strideptr[0] == sizeof(double)) &&
                (strideptr[1] == sizeof(double))) {
                // printf("Contiguous inputs; going fast\n");
                double* din1  = (double*)(dataptrarray[0]);
                double* din2  = (double*)(dataptrarray[1]);
                double* dout1 = (double*)(dataptrarray[2]);
                double* dout2 = (double*)(dataptrarray[3]);
                char* ok = dataptrarray[4];
                N = *innersizeptr;
                while (N--) {
                    *ok = func(baton, *din1, *din2, dout1, dout2);
                    ok++;
                    din1++;
                    din2++;
                    dout1++;
                    dout2++;
                }
            } else {
                // printf("Non-contiguous inputs; going slow\n");
                npy_intp stride1 = strideptr[0];
                npy_intp stride2 = strideptr[1];
                npy_intp size = *innersizeptr;
                char* src1 = dataptrarray[0];
                char* src2 = dataptrarray[1];
                double* dout1 = (double*)dataptrarray[2];
                double* dout2 = (double*)dataptrarray[3];
                char* ok = dataptrarray[4];

                for (i=0; i<size; i++) {
                    *ok = func(baton, *((double*)src1), *((double*)src2),
                               dout1, dout2);
                    ok++;
                    src1 += stride1;
                    src2 += stride2;
                    dout1++;
                    dout2++;
                }
            }
        } while (iternext(iter));

        if (PyArray_IsPythonScalar(in1) && PyArray_IsPythonScalar(in2)) {
            PyObject* px  = (PyObject*)(NpyIter_GetOperandArray(iter)[2]);
            PyObject* py  = (PyObject*)(NpyIter_GetOperandArray(iter)[3]);
            PyObject* pok = (PyObject*)(NpyIter_GetOperandArray(iter)[4]);
            //printf("Both inputs are python scalars\n");
            double d;
            unsigned char c;
            d = *(double*)PyArray_DATA((PyArrayObject*)px);
            px = PyFloat_FromDouble(d);
            d = *(double*)PyArray_DATA((PyArrayObject*)py);
            py = PyFloat_FromDouble(d);
            c = *(unsigned char*)PyArray_DATA((PyArrayObject*)pok);
            pok = PyBool_FromLong(c);
            ret = Py_BuildValue("(NNN)", pok, px, py);
            /*
             // I couldn't figure this out -- ScalarAsCtype didn't work
             if (PyArray_CheckScalar(px)) {
             printf("x is scalar\n");
             }
             if (PyArray_IsScalar(px, Double)) {
             printf("x is PyDoubleArrType\n");
             }
             if (PyArray_IsScalar(px, CDouble)) {
             printf("x is PyCDoubleArrType\n");
             }
             if (PyArray_ISFLOAT(px)) {
             printf("x ISFLOAT\n");
             }
             //PyArray_ScalarAsCtype(px, &d);
             */
        } else {
            ret = Py_BuildValue("(OOO)",
                                NpyIter_GetOperandArray(iter)[4],
                                NpyIter_GetOperandArray(iter)[2],
                                NpyIter_GetOperandArray(iter)[3]);
        }

        cleanup:
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(op[0]);
        Py_DECREF(op[1]);
        return ret;
    }


    static PyObject* broadcast_2to2i
        (
         //int func(const void*, double, double, double*, double*),
         f_2to2i func,
         const void* baton,
         PyObject* in1, PyObject* in2) {

        NpyIter *iter = NULL;
        NpyIter_IterNextFunc *iternext;
        PyArrayObject *op[5];
        PyObject *ret;
        npy_uint32 flags;
        npy_uint32 op_flags[5];
        npy_intp *innersizeptr;
        char **dataptrarray;
        npy_intp* strideptr;
        PyArray_Descr* dtypes[5];
        int j;

        // we'll do the inner loop ourselves
        flags = NPY_ITER_EXTERNAL_LOOP;
        // use buffers to satisfy dtype casts
        flags |= NPY_ITER_BUFFERED;
        // grow inner loop
        flags |= NPY_ITER_GROWINNER;

        op[0] = (PyArrayObject*)PyArray_FromAny(in1, NULL, 0, 0, 0, NULL);
        op[1] = (PyArrayObject*)PyArray_FromAny(in2, NULL, 0, 0, 0, NULL);
        // automatically allocate the output arrays.
        op[2] = NULL;
        op[3] = NULL;
        op[4] = NULL;

        if ((PyArray_Size((PyObject*)op[0]) == 0) ||
            (PyArray_Size((PyObject*)op[1]) == 0)) {
            // empty inputs -- empty outputs
            npy_intp dim = 0;
            ret = Py_BuildValue("(NNN)",
                                PyArray_SimpleNew(1, &dim, NPY_INT),
                                PyArray_SimpleNew(1, &dim, NPY_DOUBLE),
                                PyArray_SimpleNew(1, &dim, NPY_DOUBLE));
            goto cleanup;
        }

        op_flags[0] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_flags[1] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO | NPY_ITER_CONTIG | NPY_ITER_ALIGNED;
        op_flags[3] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO | NPY_ITER_CONTIG | NPY_ITER_ALIGNED;
        op_flags[4] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO | NPY_ITER_CONTIG | NPY_ITER_ALIGNED;

        dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[3] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[4] = PyArray_DescrFromType(NPY_INT);

        iter = NpyIter_MultiNew(5, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                                op_flags, dtypes);
        for (j=0; j<5; j++)
            Py_DECREF(dtypes[j]);

        if (!iter)
            return NULL;

        iternext = NpyIter_GetIterNext(iter, NULL);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        // The inner loop size and data pointers may change during the
        // loop, so just cache the addresses.
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        dataptrarray = NpyIter_GetDataPtrArray(iter);

        do {
            npy_intp i, N;
            char* src1 = dataptrarray[0];
            char* src2 = dataptrarray[1];
            double* dout1 = (double*)(dataptrarray[2]);
            double* dout2 = (double*)(dataptrarray[3]);
            int* ok = (int*)dataptrarray[4];
            N = *innersizeptr;

            //printf("2to2i: N=%i, strides %i,%i\n", N, strideptr[0], strideptr[1]);

            // are the inputs contiguous?  (Outputs will be, since we
            // allocated them)
            if ((strideptr[0] == sizeof(double)) &&
                (strideptr[1] == sizeof(double))) {
                // printf("Contiguous inputs; going fast\n");
                double* din1  = (double*)src1;
                double* din2  = (double*)src2;
                while (N--) {
                    *ok = func(baton, *din1, *din2, dout1, dout2);
                    ok++;
                    din1++;
                    din2++;
                    dout1++;
                    dout2++;
                }
            } else {
                // printf("Non-contiguous inputs; going slow\n");
                npy_intp stride1 = strideptr[0];
                npy_intp stride2 = strideptr[1];
                for (i=0; i<N; i++) {
                    *ok = func(baton, *((double*)src1), *((double*)src2),
                               dout1, dout2);
                    ok++;
                    src1 += stride1;
                    src2 += stride2;
                    dout1++;
                    dout2++;
                }
            }
        } while (iternext(iter));

        if (PyArray_IsPythonScalar(in1) && PyArray_IsPythonScalar(in2)) {
            PyObject* px  = (PyObject*)(NpyIter_GetOperandArray(iter)[2]);
            PyObject* py  = (PyObject*)(NpyIter_GetOperandArray(iter)[3]);
            PyObject* pok = (PyObject*)(NpyIter_GetOperandArray(iter)[4]);
            //printf("Both inputs are python scalars\n");
            double d;
            int i;
            d = *(double*)PyArray_DATA((PyArrayObject*)px);
            px = PyFloat_FromDouble(d);
            d = *(double*)PyArray_DATA((PyArrayObject*)py);
            py = PyFloat_FromDouble(d);
            i = *(int*)PyArray_DATA((PyArrayObject*)pok);
            pok = PyInt_FromLong(i);
            ret = Py_BuildValue("(NNN)", pok, px, py);
        } else {
            // Grab the results -- note "4,2,3" order -- ok,x,y
            ret = Py_BuildValue("(OOO)",
                                NpyIter_GetOperandArray(iter)[4],
                                NpyIter_GetOperandArray(iter)[2],
                                NpyIter_GetOperandArray(iter)[3]);
        }
        cleanup:
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(op[0]);
        Py_DECREF(op[1]);
        return ret;
    }

    static PyObject* broadcast_2to2
        (
         //void func(const void*, double, double, double*, double*),
         f_2to2 func,
         const void* baton,
         PyObject* in1, PyObject* in2) {

        NpyIter *iter = NULL;
        NpyIter_IterNextFunc *iternext;
        PyArrayObject *op[4];
        PyObject *ret;
        npy_uint32 flags;
        npy_uint32 op_flags[4];
        npy_intp *innersizeptr;
        char **dataptrarray;
        npy_intp* strideptr;
        PyArray_Descr* dtypes[4];
        npy_intp i;

        // we'll do the inner loop ourselves
        flags = NPY_ITER_EXTERNAL_LOOP;
        // use buffers to satisfy dtype casts
        flags |= NPY_ITER_BUFFERED;
        // grow inner loop
        flags |= NPY_ITER_GROWINNER;

        op[0] = (PyArrayObject*)PyArray_FromAny(in1, NULL, 0, 0, 0, NULL);
        op[1] = (PyArrayObject*)PyArray_FromAny(in2, NULL, 0, 0, 0, NULL);
        // automatically allocate the output arrays.
        op[2] = NULL;
        op[3] = NULL;

        if ((PyArray_Size((PyObject*)op[0]) == 0) ||
            (PyArray_Size((PyObject*)op[1]) == 0)) {
            // empty inputs -- empty outputs
            npy_intp dim = 0;
            ret = Py_BuildValue("(NN)",
                                PyArray_SimpleNew(1, &dim, NPY_DOUBLE),
                                PyArray_SimpleNew(1, &dim, NPY_DOUBLE));
            goto cleanup;
        }

        op_flags[0] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_flags[1] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO;
        op_flags[3] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO;

        dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
        dtypes[3] = PyArray_DescrFromType(NPY_DOUBLE);

        iter = NpyIter_MultiNew(4, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                                op_flags, dtypes);
        for (i=0; i<4; i++)
            Py_DECREF(dtypes[i]);
        if (!iter)
            return NULL;

        iternext = NpyIter_GetIterNext(iter, NULL);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        // The inner loop size and data pointers may change during the
        // loop, so just cache the addresses.
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        dataptrarray = NpyIter_GetDataPtrArray(iter);

        do {
            // are the inputs contiguous?  (Outputs will be, since we
            // allocated them)
            if ((strideptr[0] == sizeof(double)) &&
                (strideptr[1] == sizeof(double))) {

                npy_intp N = *innersizeptr;
                double* din1  = (double*)(dataptrarray[0]);
                double* din2  = (double*)(dataptrarray[1]);
                double* dout1 = (double*)(dataptrarray[2]);
                double* dout2 = (double*)(dataptrarray[3]);

                //printf("Contiguous inputs; going fast\n");
                //printf("Inner loop: %i\n", (int)N);
                //printf("Output strides: %i %i\n", (int)strideptr[2], (int)strideptr[3]);
                //printf("Strides: %i %i %i %i\n", (int)strideptr[0], (int)strideptr[1], (int)strideptr[2], (int)strideptr[3]);

                while (N--) {
                    //printf("Calling %i: inputs (%12g,%12g)\n", (int)N, *din1, *din2);
                    func(baton, *din1, *din2, dout1, dout2);
                    din1++;
                    din2++;
                    dout1++;
                    dout2++;
                }
            } else {
                npy_intp stride1 = NpyIter_GetInnerStrideArray(iter)[0];
                npy_intp stride2 = NpyIter_GetInnerStrideArray(iter)[1];
                npy_intp size = *innersizeptr;
                char*   src1  = dataptrarray[0];
                char*   src2  = dataptrarray[1];
                double* dout1 = (double*)(dataptrarray[2]);
                double* dout2 = (double*)(dataptrarray[3]);

                //printf("Non-contiguous inputs; going slow\n");
                //printf("%i items\n", (int)size);

                for (i=0; i<size; i++) {
                    //printf("Call %i: inputs (%12g,%12g)\n", (int)i, ((double*)src1)[0], ((double*)src2)[0]);
                    func(baton, *((double*)src1), *((double*)src2),
                         dout1, dout2);
                    src1 += stride1;
                    src2 += stride2;
                    dout1++;
                    dout2++;
                }
            }
        } while (iternext(iter));

        if (PyArray_IsPythonScalar(in1) && PyArray_IsPythonScalar(in2)) {
            PyObject* px  = (PyObject*)(NpyIter_GetOperandArray(iter)[2]);
            PyObject* py  = (PyObject*)(NpyIter_GetOperandArray(iter)[3]);
            //printf("Both inputs are python scalars\n");
            double d;
            d = *(double*)PyArray_DATA((PyArrayObject*)px);
            px = PyFloat_FromDouble(d);
            d = *(double*)PyArray_DATA((PyArrayObject*)py);
            py = PyFloat_FromDouble(d);
            ret = Py_BuildValue("(NN)", px, py);
        } else {
            // Grab the results
            ret = Py_BuildValue("(OO)",
                                NpyIter_GetOperandArray(iter)[2],
                                NpyIter_GetOperandArray(iter)[3]);
        }

        cleanup:
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(op[0]);
        Py_DECREF(op[1]);
        return ret;
    }

    static int tan_wcs_resample(tan_t* inwcs, tan_t* outwcs,
                                PyObject* py_inimg, PyObject* py_outimg,
                                int weighted, int lorder) {
        PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
        int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
        int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_UPDATEIFCOPY;
        PyArrayObject *np_inimg=NULL, *np_outimg=NULL;

        Py_INCREF(dtype);
        Py_INCREF(dtype);
        np_inimg  = (PyArrayObject*)PyArray_CheckFromAny(py_inimg,  dtype, 2, 2, req, NULL);
        np_outimg = (PyArrayObject*)PyArray_CheckFromAny(py_outimg, dtype, 2, 2, reqout, NULL);
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

    static int tan_numpy_xyz2pixelxy(tan_t* tan, PyArrayObject* npxyz,
                                     PyArrayObject* npx, PyArrayObject* npy) {
        npy_intp i, N;
        int rtn = 0;
        double *x, *y;

        if (PyArray_NDIM(npx) != 1) {
            PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional");
            return -1;
        }
        if (PyArray_TYPE(npx) != NPY_DOUBLE) {
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


    static int an_tally(PyObject* py_counts, PyObject* py_x, PyObject* py_y) {
        PyArray_Descr* itype;
        PyArrayObject *np_counts=NULL, *np_x=NULL, *np_y=NULL;
        int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED |
            NPY_ARRAY_ELEMENTSTRIDES;
        int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_UPDATEIFCOPY;
        int32_t *counts, *px, *py;
        int W, H, i, N;

        itype = PyArray_DescrFromType(NPY_INT32);
        Py_INCREF(itype);
        Py_INCREF(itype);

        np_counts = (PyArrayObject*)PyArray_CheckFromAny(py_counts, itype, 2, 2, reqout, NULL);
        np_x = (PyArrayObject*)PyArray_CheckFromAny(py_x, itype, 1, 1, req, NULL);
        np_y = (PyArrayObject*)PyArray_CheckFromAny(py_y, itype, 1, 1, req, NULL);

        if (!np_counts || !np_x || !np_y) {
            ERR("Failed to PyArray_FromAny the counts, x, and y arrays.\n");
            Py_XDECREF(np_counts);
            Py_XDECREF(np_x);
            Py_XDECREF(np_y);
            return -1;
        }
        N = (int)PyArray_DIM(np_x, 0);
        if (PyArray_DIM(np_y, 0) != N) {
            ERR("Expected x and y arrays to have the same lengths!\n");
            Py_XDECREF(np_counts);
            Py_XDECREF(np_x);
            Py_XDECREF(np_y);
            return -1;
        }

        H = (int)PyArray_DIM(np_counts, 0);
        W = (int)PyArray_DIM(np_counts, 1);
        //printf("Counts array size %i x %i\n", W, H);
        counts = PyArray_DATA(np_counts);
        px = PyArray_DATA(np_x);
        py = PyArray_DATA(np_y);
        for (i=0; i<N; i++) {
            int32_t xi = (*px);
            int32_t yi = (*py);
            if (yi < 0 || yi >= H || xi < 0 || xi >= W) {
                printf("Warning: skipping out-of-range value: i=%i, xi,yi = %i,%i\n", i, xi, yi);
            } else {
                counts[yi*W + xi]++;
            }
            px++;
            py++;
        }
        Py_DECREF(np_counts);
        Py_DECREF(np_x);
        Py_DECREF(np_y);
        return 0;
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

def tan_t_addtoheader(self, hdr):
    '''Adds this TAN WCS header to the given fitsio header'''
    hdr.add_record(dict(name='CTYPE1', value='RA---TAN', comment='TANgent plane'))
    hdr.add_record(dict(name='CTYPE2', value='DEC--TAN', comment='TANgent plane'))
    hdr.add_record(dict(name='CRVAL1', value=self.crval[0], comment='Reference RA'))
    hdr.add_record(dict(name='CRVAL2', value=self.crval[1], comment='Reference Dec'))
    hdr.add_record(dict(name='CRPIX1', value=self.crpix[0], comment='Reference x'))
    hdr.add_record(dict(name='CRPIX2', value=self.crpix[1], comment='Reference y'))
    hdr.add_record(dict(name='CD1_1', value=self.cd[0], comment='CD matrix'))
    hdr.add_record(dict(name='CD1_2', value=self.cd[1], comment='CD matrix'))
    hdr.add_record(dict(name='CD2_1', value=self.cd[2], comment='CD matrix'))
    hdr.add_record(dict(name='CD2_2', value=self.cd[3], comment='CD matrix'))
    hdr.add_record(dict(name='IMAGEW', value=self.imagew, comment='Image width'))
    hdr.add_record(dict(name='IMAGEH', value=self.imageh, comment='Image height'))
tan_t.add_to_header = tan_t_addtoheader

## picklable?
def tan_t_getstate(self):
    return (self.crpix[0], self.crpix[1], self.crval[0], self.crval[1],
            self.cd[0], self.cd[1], self.cd[2], self.cd[3],
            self.imagew, self.imageh, self.sin)
def tan_t_setstate(self, state):
    #print 'setstate: self', self, 'state', state
    #print 'state', state
    self.this = _util.new_tan_t()
    #print 'self', repr(self)
    p0,p1,v0,v1,cd0,cd1,cd2,cd3,w,h,sin = state
    self.set_crpix(p0,p1)
    self.set_crval(v0,v1)
    self.set_cd(cd0,cd1,cd2,cd3)
    self.set_imagesize(w,h)
    self.sin = sin
    #(self.crpix[0], self.crpix[1], self.crval[0], self.crval[1],
    #self.cd[0], self.cd[1], self.cd[2], self.cd[3],
    #self.imagew, self.imageh) = state
def tan_t_getnewargs(self):
    return ()
tan_t.__getstate__ = tan_t_getstate
tan_t.__setstate__ = tan_t_setstate
tan_t.__getnewargs__ = tan_t_getnewargs

def tan_t_getshape(self):
    return int(self.imageh), int(self.imagew)

tan_t.shape = property(tan_t_getshape)

def tan_t_get_cd(self):
    cd = self.cd
    return (cd[0], cd[1], cd[2], cd[3])
tan_t.get_cd = tan_t_get_cd

def tan_t_pixelxy2radec(self, x, y):
    return tan_xy2rd_wrapper(self.this, x, y)
tan_t.pixelxy2radec_single = tan_t.pixelxy2radec
tan_t.pixelxy2radec = tan_t_pixelxy2radec

def tan_t_radec2pixelxy(self, r, d):
    return tan_rd2xy_wrapper(self.this, r, d)
tan_t.radec2pixelxy_single = tan_t.radec2pixelxy
tan_t.radec2pixelxy = tan_t_radec2pixelxy

def tan_t_iwc2pixelxy(self, r, d):
    return tan_iwc2xy_wrapper(self.this, r, d)
tan_t.iwc2pixelxy_single = tan_t.iwc2pixelxy
tan_t.iwc2pixelxy = tan_t_iwc2pixelxy

def tan_t_pixelxy2iwc(self, x,y):
    return tan_xy2iwc_wrapper(self.this, x,y)
tan_t.pixelxy2iwc_single = tan_t.pixelxy2iwc
tan_t.pixelxy2iwc = tan_t_pixelxy2iwc

def tan_t_radec2iwc(self, r, d):
    return tan_rd2iwc_wrapper(self.this, r, d)
tan_t.radec2iwc_single = tan_t.radec2iwc
tan_t.radec2iwc = tan_t_radec2iwc

def tan_t_iwc2radec(self, u, v):
    return tan_iwc2rd_wrapper(self.this, u, v)
tan_t.iwc2radec_single = tan_t.iwc2radec
tan_t.iwc2radec = tan_t_iwc2radec

def sip_t_pixelxy2radec(self, x, y):
    return sip_xy2rd_wrapper(self.this, x, y)
sip_t.pixelxy2radec_single = sip_t.pixelxy2radec
sip_t.pixelxy2radec = sip_t_pixelxy2radec

def sip_t_radec2pixelxy(self, r, d):
    return sip_rd2xy_wrapper(self.this, r, d)
sip_t.radec2pixelxy_single = sip_t.radec2pixelxy
sip_t.radec2pixelxy = sip_t_radec2pixelxy

def sip_t_iwc2pixelxy(self, r, d):
    return sip_iwc2xy_wrapper(self.this, r, d)
sip_t.iwc2pixelxy_single = sip_t.iwc2pixelxy
sip_t.iwc2pixelxy = sip_t_iwc2pixelxy

def sip_t_pixelxy2iwc(self, x,y):
    return sip_xy2iwc_wrapper(self.this, x,y)
sip_t.pixelxy2iwc_single = sip_t.pixelxy2iwc
sip_t.pixelxy2iwc = sip_t_pixelxy2iwc

def sip_t_radec2iwc(self, r, d):
    return sip_rd2iwc_wrapper(self.this, r, d)
sip_t.radec2iwc_single = sip_t.radec2iwc
sip_t.radec2iwc = sip_t_radec2iwc

def sip_t_iwc2radec(self, u, v):
    return sip_iwc2rd_wrapper(self.this, u, v)
sip_t.iwc2radec_single = sip_t.iwc2radec
sip_t.iwc2radec = sip_t_iwc2radec


def anwcs_t_pixelxy2radec(self, x, y):
    ok,r,d =  anwcs_xy2rd_wrapper(self.this, x, y)
    return (ok == 0),r,d
anwcs_t.pixelxy2radec_single = anwcs_t.pixelxy2radec
anwcs_t.pixelxy2radec = anwcs_t_pixelxy2radec

def anwcs_t_radec2pixelxy(self, r, d):
    ok,x,y =  anwcs_rd2xy_wrapper(self.this, r, d)
    return (ok == 0),x,y
anwcs_t.radec2pixelxy_single = anwcs_t.radec2pixelxy
anwcs_t.radec2pixelxy = anwcs_t_radec2pixelxy

def tan_t_radec_bounds(self):
    W,H = self.imagew, self.imageh
    r,d = self.pixelxy2radec([1, W/2, W, W, W, W/2, 1, 1], [1, 1, 1, H/2, H, H, H, H/2])
    rx = r.max()
    rn = r.min()
    # ugh, RA wrap-around.  We find the largest value < 180 (ie, near zero) and smallest value > 180 (ie, near 360)
    # and report them with ralo > rahi so that this case can be identified
    if rx - rn > 180:
        rx = r[r < 180].max()
        rn = r[r > 180].min()
    return (rn, rx, d.min(), d.max())
tan_t.radec_bounds = tan_t_radec_bounds

_real_tan_t_init = tan_t.__init__
def my_tan_t_init(self, *args, **kwargs):
    # fitsio header: check for '.records()' function.
    if len(args) == 1 and hasattr(args[0], 'records'):
        try:
            hdr = args[0]
            qhdr = fitsio_to_qfits_header(hdr)
            args = [qhdr]
        except:
            pass

    _real_tan_t_init(self, *args, **kwargs)
    if self.this is None:
        raise RuntimeError('Duck punch!')
tan_t.__init__ = my_tan_t_init

Tan = tan_t

def tan_t_get_subimage(self, x0, y0, w, h):
    wcs2 = tan_t(self)
    cpx,cpy = wcs2.crpix
    wcs2.set_crpix(cpx - x0, cpy - y0)
    wcs2.set_width(float(w))
    wcs2.set_height(float(h))
    return wcs2
tan_t.get_subimage = tan_t_get_subimage

# Deja Vu!
# def sip_t_get_subimage(self, xlo, xhi, ylo, yhi):
#     sipout = sip_t(self)
#     sip_shift(self.this, sipout.this, float(xlo), float(xhi), float(ylo), float(yhi))
#     return sipout
# sip_t.get_subimage = sip_t_get_subimage

# picklable
def sip_t_getstate(self):
    t = (self.wcstan.__getstate__(),
         self.a_order, self.b_order, self.a, self.b,
         self.ap_order, self.bp_order, self.ap, self.bp)
    return t

def sip_t_setstate(self, s):
    self.this = _util.new_sip_t()
    (t, self.a_order, self.b_order, self.a, self.b,
     self.ap_order, self.bp_order, self.ap, self.bp) = s
    #self.wcstan.__setstate__(t)
    # disturbingly, tan_t_setstate does not work because it resets self.this = ...
    p0,p1,v0,v1,cd0,cd1,cd2,cd3,w,h,sin = t
    self.wcstan.set_crpix(p0,p1)
    self.wcstan.set_crval(v0,v1)
    self.wcstan.set_cd(cd0,cd1,cd2,cd3)
    self.wcstan.set_imagesize(w,h)
    self.wcstan.sin = sin

def sip_t_getnewargs(self):
    return ()

sip_t.__getstate__ = sip_t_getstate
sip_t.__setstate__ = sip_t_setstate
sip_t.__getnewargs__ = sip_t_getnewargs

%}

%include "fitsioutils.h"

// dcen3x3
%apply float *OUTPUT { float *xcen, float *ycen };

%include "dimage.h"

%inline %{
int dcen3x3b(float i0, float i1, float i2,
             float i3, float i4, float i5,
             float i6, float i7, float i8,
             float *xcen, float *ycen) {
float im[9];
im[0] = i0;
im[1] = i1;
im[2] = i2;
im[3] = i3;
im[4] = i4;
im[5] = i5;
im[6] = i6;
im[7] = i7;
im[8] = i8;
return dcen3x3(im, xcen, ycen);
}
%}
