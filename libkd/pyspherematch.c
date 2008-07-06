// /opt/local/include/python2.5/
#include "Python.h"

#include <stdio.h>

// PYTHON/site-packages/numpy/core/include
#include "arrayobject.h"
//#include "ufuncobject.h"
//#include "log.h"

#include "kdtree.h"

static void info(PyArrayObject* x);

static PyObject* spherematch_info(PyObject* self, PyObject* args) {
    //int result;
    int N, D;
    int i,j;
    int Nleaf, treeoptions, treetype;
    kdtree_t* kd;
    double* data;
    //const char *command;
    PyArrayObject* x;

    //if (!PyArg_ParseTuple(args, "s", &command))
    //if (!PyArg_ParseTuple(args, .i:function.,&x))
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x))
        return NULL;

    if (PyArray_NDIM(x) != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "array must be two-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(x) != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,
                        "array must contain floats");
        return NULL;
    }

    N = PyArray_DIM(x, 0);
    D = PyArray_DIM(x, 1);

    if (D > 10) {
        PyErr_SetString(PyExc_ValueError,
                        "maximum dimensionality is 10: maybe you need to transpose your array?");
        return NULL;
    }

    for (i=0; i<N; i++) {
        printf("data pt = [ ");
        for (j=0; j<D; j++) {
            double* pd = PyArray_GETPTR2(x, i, j);
            double xx = *pd;
            printf("%g ", xx);
        }
        printf("]\n");
    }

    info(x);

    Nleaf = 16;
    treetype = KDTT_DOUBLE;
    treeoptions = KD_BUILD_SPLIT;

    data = malloc(N * D * sizeof(double));
    for (i=0; i<N; i++) {
        for (j=0; j<D; j++) {
            double* pd = PyArray_GETPTR2(x, i, j);
            data[i*D + j] = *pd;
        }
    }

    kd = kdtree_build(NULL, data, N, D, Nleaf,
                      treetype, treeoptions);

    return Py_BuildValue("k", kd);
}

static PyMethodDef spherematchMethods[] = {
    { "spherematch_info", spherematch_info, METH_VARARGS,
      "get info about an array" },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initspherematch(void) {
    Py_InitModule("spherematch", spherematchMethods);

    import_array();
}

static void info(PyArrayObject* x) {
    int i;
    int D;
    printf("data %p\n", PyArray_DATA(x));
    //PyArray_BYTES(x);
    printf("itemsize %i\n", PyArray_ITEMSIZE(x));
    printf("ndim %i\n", PyArray_NDIM(x));
    D = PyArray_NDIM(x);
    //PyArray_DIMS(x);
    for (i=0; i<D; i++)
        printf("  dim %i: %li\n", i, PyArray_DIM(x, i));
    //PyArray_STRIDES(x);
    for (i=0; i<D; i++)
        printf("  stride %i: %li\n", i, PyArray_STRIDE(x,i));
    //PyArray_DESCR(x);
    //PyArray_BASE(x);

    /*
     PyArrayIterObject it;

     PyArray_ITER_NEXT(it);
     PyArray_ITER_DATA(it);
     
     */
}

