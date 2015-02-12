%module(package="anresample") _resample

%include <typemaps.i>
%include <cstring.i>
%include <exception.i>

%{
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <sys/param.h>
#include <stdlib.h>
%}

%init %{
      // numpy
      import_array();
%}

%inline %{
    #define ERR(x, ...) printf(x, ## __VA_ARGS__)

    #define LANCZOS_INTERP_FUNC lanczos5_interpolate
    #define L 5
        static int LANCZOS_INTERP_FUNC(PyObject* np_ixi, PyObject* np_iyi,
                                       PyObject* np_dx, PyObject* np_dy,
                                       PyObject* loutputs, PyObject* linputs);
    #include "lanczos.c"
    #undef LANCZOS_INTERP_FUNC
    #undef L

    #define LANCZOS_INTERP_FUNC lanczos3_interpolate
    #define L 3
        static int LANCZOS_INTERP_FUNC(PyObject* np_ixi, PyObject* np_iyi,
                                       PyObject* np_dx, PyObject* np_dy,
                                       PyObject* loutputs, PyObject* linputs);
    #include "lanczos.c"
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

    
%}
