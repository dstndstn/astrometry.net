%module(package="astrometry.util") util

%include <typemaps.i>
%include <cstring.i>
%include <exception.i>

%{
// numpy.
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <sys/param.h>
#include <stdlib.h>

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
#define ERR(x, ...)                             \
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


    static double flat_median_f(PyObject* np_arr) {
        PyArray_Descr* dtype;
        npy_intp N;
        int req = NPY_C_CONTIGUOUS | NPY_ALIGNED |
            NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;
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
        memcpy(x, PyArray_DATA(np_arr), sizeof(float)*N);
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
        mid = (int)(N/2);
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

    static int median_smooth(PyObject* np_image,
                             PyObject* np_mask,
                             int halfbox,
                             PyObject* np_smooth) {
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
            return -1;
        }
        if (np_mask != Py_None) {
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

        if (np_mask != Py_None) {
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

    #define LANCZOS_INTERP_FUNC lanczos5_interpolate
    #define L 5
        static int LANCZOS_INTERP_FUNC(PyObject* np_ixi, PyObject* np_iyi,
                                       PyObject* np_dx, PyObject* np_dy,
                                       PyObject* loutputs, PyObject* linputs);
    #include "lanczos.i"
    #undef LANCZOS_INTERP_FUNC
    #undef L

    #define LANCZOS_INTERP_FUNC lanczos3_interpolate
    #define L 3
        static int LANCZOS_INTERP_FUNC(PyObject* np_ixi, PyObject* np_iyi,
                                       PyObject* np_dx, PyObject* np_dy,
                                       PyObject* loutputs, PyObject* linputs);
    #include "lanczos.i"
    #undef LANCZOS_INTERP_FUNC
    #undef L

    static int lanczos3_filter(PyObject* np_dx, PyObject* np_f) {
        npy_intp N;
        npy_intp i;
        float* dx;
        float* f;

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

    static int lanczos3_filter_table(PyObject* np_dx, PyObject* np_f, int rangecheck) {
        npy_intp N;
        npy_intp i;
        float* dx;
        float* f;

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
            ERR("sizeof dx %i\n", PyArray_ITEMSIZE(np_dx));
        }
        if ((PyArray_ITEMSIZE(np_f ) != sizeof(float))) {
            ERR("sizeof f %i\n", PyArray_ITEMSIZE(np_f));
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






    static int lanczos_shift_image_c(PyObject* np_img, PyObject* np_weight,
                                     PyObject* np_outimg,
                                     PyObject* np_outweight,
                                     int order, double dx, double dy) {
        int W,H;
        int i,j;

        lanczos_args_t lanczos;

        PyArray_Descr* dtype;
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

	dtype = PyArray_DescrFromType(PyArray_DOUBLE);
	Py_INCREF(dtype);
        np_img = PyArray_CheckFromAny(np_img, dtype, 2, 2, req, NULL);
        if (np_weight != Py_None) {
	    Py_INCREF(dtype);
            np_weight = PyArray_CheckFromAny(np_weight, dtype, 2, 2, req, NULL);
            if (!np_weight) {
                ERR("Failed to run PyArray_FromAny on np_weight\n");
                return -1;
            }
        }
	Py_INCREF(dtype);
        np_outimg = PyArray_CheckFromAny(np_outimg, dtype, 2, 2, reqout, NULL);
        if (np_outweight != Py_None) {
	    Py_INCREF(dtype);
            np_outweight = PyArray_CheckFromAny(np_outweight, dtype, 2, 2, reqout, NULL);
        }
	Py_DECREF(dtype);
	dtype = NULL;

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

    // healpix_radec_bounds
%apply double *OUTPUT { double *ralo, double *rahi, double *declo, double *dechi };

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

def anwcs_get_cd(self):
    return anwcs_get_cd_matrix(self)
anwcs.get_cd = anwcs_get_cd

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
    void iwc2pixelxy(double u, double v, double *p_x, double *p_y) {
        sip_iwc2pixelxy($self, u, v, p_x, p_y);
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

# def sip_t_get_subimage(self, x0, y0, w, h):
#     wcs2 = sip_t(self)
#     cpx,cpy = wcs2.crpix
#     wcs2.set_crpix((cpx - x0, cpy - y0))
#     wcs2.set_width(float(w))
#     wcs2.set_height(float(h))
#     return wcs2
# sip_t.get_subimage = sip_t_get_subimage

sip_t.imagew = property(sip_t.get_width,  sip_t.set_width,  None, 'image width')
sip_t.imageh = property(sip_t.get_height, sip_t.set_height, None, 'image height')

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
    W,H = self.wcstan.imagew, self.wcstan.imageh
    r,d = self.pixelxy2radec([1, W, W, 1], [1, 1, H, H])
    return (r.min(), r.max(), d.min(), d.max())
sip_t.radec_bounds = sip_t_radec_bounds    

#def sip_t_fromstring(s):
#   sip = sip_from_string(s, len(s),

_real_sip_t_init = sip_t.__init__
def my_sip_t_init(self, *args, **kwargs):
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
                             PyObject* np_img, PyObject* np_weight,
                             float fweight, const anwcs_t* wcs) {
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
    // in numpy v2.0 these constants have a NPY_ARRAY_ prefix
    int req = NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;
    float *img, *weight;

    Py_INCREF(dtype);
    np_img = PyArray_CheckFromAny(np_img, dtype, 2, 2, req, NULL);
    img = PyArray_DATA(np_img);
    if (!np_img) {
      ERR("Failed to PyArray_FromAny the image\n");
      Py_XDECREF(np_img);
      Py_DECREF(dtype);
      return -1;
    }
    if (np_weight == Py_None) {
      weight = NULL;
    } else {
      Py_INCREF(dtype);
      np_weight = PyArray_CheckFromAny(np_weight, dtype, 2, 2, req, NULL);
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
    coadd_get_snapshot(co, PyArray_DATA(npimg), badpix);
    return npimg;
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

        // are the inputs contiguous?  (Outputs will be, since we
        // allocated them)
        if ((strideptr[0] == sizeof(double)) &&
            (strideptr[1] == sizeof(double))) {
            // printf("Contiguous inputs; going fast\n");
            do {
                N = *innersizeptr;
                double* din1 = (double*)dataptrarray[0];
                double* din2 = (double*)dataptrarray[1];
                double* dout1 = (double*)dataptrarray[2];
                double* dout2 = (double*)dataptrarray[3];
                char* ok = dataptrarray[4];
                while (N--) {
                    *ok = func(baton, *din1, *din2, dout1, dout2);
                    ok++;
                    din1++;
                    din2++;
                    dout1++;
                    dout2++;
                }
            } while (iternext(iter));
        } else {
            // printf("Non-contiguous inputs; going slow\n");
            npy_intp stride1 = NpyIter_GetInnerStrideArray(iter)[0];
            npy_intp stride2 = NpyIter_GetInnerStrideArray(iter)[1];
            do {
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
            } while (iternext(iter));
        }

        if (PyArray_IsPythonScalar(in1) && PyArray_IsPythonScalar(in2)) {
            PyObject* px  = (PyObject*)NpyIter_GetOperandArray(iter)[2];
            PyObject* py  = (PyObject*)NpyIter_GetOperandArray(iter)[3];
            PyObject* pok = (PyObject*)NpyIter_GetOperandArray(iter)[4];
            //printf("Both inputs are python scalars\n");
            double d;
            unsigned char c;
            d = *(double*)PyArray_DATA(px);
            px = PyFloat_FromDouble(d);
            d = *(double*)PyArray_DATA(py);
            py = PyFloat_FromDouble(d);
            c = *(unsigned char*)PyArray_DATA(pok);
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
                                PyArray_SimpleNew(1, &dim, NPY_INT),
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
        dtypes[4] = PyArray_DescrFromType(NPY_INT);

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

        // are the inputs contiguous?  (Outputs will be, since we
        // allocated them)
        if ((strideptr[0] == sizeof(double)) &&
            (strideptr[1] == sizeof(double))) {
            // printf("Contiguous inputs; going fast\n");
            do {
                N = *innersizeptr;
                double* din1 = (double*)dataptrarray[0];
                double* din2 = (double*)dataptrarray[1];
                double* dout1 = (double*)dataptrarray[2];
                double* dout2 = (double*)dataptrarray[3];
                int* ok = (int*)dataptrarray[4];
                while (N--) {
                    *ok = func(baton, *din1, *din2, dout1, dout2);
                    ok++;
                    din1++;
                    din2++;
                    dout1++;
                    dout2++;
                }
            } while (iternext(iter));
        } else {
            // printf("Non-contiguous inputs; going slow\n");
            npy_intp stride1 = NpyIter_GetInnerStrideArray(iter)[0];
            npy_intp stride2 = NpyIter_GetInnerStrideArray(iter)[1];
            do {
                npy_intp size = *innersizeptr;
                char* src1 = dataptrarray[0];
                char* src2 = dataptrarray[1];
                double* dout1 = (double*)dataptrarray[2];
                double* dout2 = (double*)dataptrarray[3];
                int* ok = (int*)dataptrarray[4];

                for (i=0; i<size; i++) {
                    *ok = func(baton, *((double*)src1), *((double*)src2),
                               dout1, dout2);
                    ok++;
                    src1 += stride1;
                    src2 += stride2;
                    dout1++;
                    dout2++;
                }
            } while (iternext(iter));
        }

        if (PyArray_IsPythonScalar(in1) && PyArray_IsPythonScalar(in2)) {
            PyObject* px  = (PyObject*)NpyIter_GetOperandArray(iter)[2];
            PyObject* py  = (PyObject*)NpyIter_GetOperandArray(iter)[3];
            PyObject* pok = (PyObject*)NpyIter_GetOperandArray(iter)[4];
            //printf("Both inputs are python scalars\n");
            double d;
            int i;
            d = *(double*)PyArray_DATA(px);
            px = PyFloat_FromDouble(d);
            d = *(double*)PyArray_DATA(py);
            py = PyFloat_FromDouble(d);
            i = *(int*)PyArray_DATA(pok);
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

        // are the inputs contiguous?  (Outputs will be, since we
        // allocated them)
        if ((strideptr[0] == sizeof(double)) &&
            (strideptr[1] == sizeof(double))) {
            // printf("Contiguous inputs; going fast\n");
            do {
                N = *innersizeptr;
                double* din1 = (double*)dataptrarray[0];
                double* din2 = (double*)dataptrarray[1];
                double* dout1 = (double*)dataptrarray[2];
                double* dout2 = (double*)dataptrarray[3];
                while (N--) {
                    func(baton, *din1, *din2, dout1, dout2);
                    din1++;
                    din2++;
                    dout1++;
                    dout2++;
                }
            } while (iternext(iter));
        } else {
            // printf("Non-contiguous inputs; going slow\n");
            npy_intp stride1 = NpyIter_GetInnerStrideArray(iter)[0];
            npy_intp stride2 = NpyIter_GetInnerStrideArray(iter)[1];
            do {
                npy_intp size = *innersizeptr;
                char* src1 = dataptrarray[0];
                char* src2 = dataptrarray[1];
                double* dout1 = (double*)dataptrarray[2];
                double* dout2 = (double*)dataptrarray[3];
                for (i=0; i<size; i++) {
                    func(baton, *((double*)src1), *((double*)src2),
                         dout1, dout2);
                    src1 += stride1;
                    src2 += stride2;
                    dout1++;
                    dout2++;
                }
            } while (iternext(iter));
        }

        if (PyArray_IsPythonScalar(in1) && PyArray_IsPythonScalar(in2)) {
            PyObject* px  = (PyObject*)NpyIter_GetOperandArray(iter)[2];
            PyObject* py  = (PyObject*)NpyIter_GetOperandArray(iter)[3];
            //printf("Both inputs are python scalars\n");
            double d;
            d = *(double*)PyArray_DATA(px);
            px = PyFloat_FromDouble(d);
            d = *(double*)PyArray_DATA(py);
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

    static int tan_numpy_xyz2pixelxy(tan_t* tan, PyObject* npxyz,
           PyObject* npx, PyObject* npy) {
        npy_intp i, N;
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

def sip_t_radec2iwc(self, r, d):
    return sip_rd2iwc_wrapper(self.this, r, d)
sip_t.radec2iwc_single = sip_t.radec2iwc
sip_t.radec2iwc = sip_t_radec2iwc

def sip_t_iwc2radec(self, u, v):
    return sip_iwc2rd_wrapper(self.this, u, v)
sip_t.iwc2radec_single = sip_t.iwc2radec
sip_t.iwc2radec = sip_t_iwc2radec


def anwcs_t_pixelxy2radec(self, x, y):
    return anwcs_xy2rd_wrapper(self.this, x, y)
anwcs_t.pixelxy2radec_single = anwcs_t.pixelxy2radec
anwcs_t.pixelxy2radec = anwcs_t_pixelxy2radec

def anwcs_t_radec2pixelxy(self, r, d):
    return anwcs_rd2xy_wrapper(self.this, r, d)
anwcs_t.radec2pixelxy_single = anwcs_t.radec2pixelxy
anwcs_t.radec2pixelxy = anwcs_t_radec2pixelxy






def tan_t_radec_bounds(self):
    W,H = self.imagew, self.imageh
    r,d = self.pixelxy2radec([1, W, W, 1], [1, 1, H, H])
    return (r.min(), r.max(), d.min(), d.max())
tan_t.radec_bounds = tan_t_radec_bounds    


_real_tan_t_init = tan_t.__init__
def my_tan_t_init(self, *args, **kwargs):
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
    # disturbingly, tan_t_setstate doesn't work because it resets self.this = ...
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
%include "dimage.h"
