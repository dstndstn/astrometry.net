#include "Python.h"

#include <stdio.h>

// PYTHON/site-packages/numpy/core/include
#include "arrayobject.h"
//#include "ufuncobject.h"

#include "kdtree.h"
#include "dualtree_rangesearch.h"
#include "bl.h"

//static void info(PyArrayObject* x);

static PyObject* spherematch_kdtree_build(PyObject* self, PyObject* args) {
    int N, D;
    int i,j;
    int Nleaf, treeoptions, treetype;
    kdtree_t* kd;
    double* data;
    PyArrayObject* x;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x))
        return NULL;

    if (PyArray_NDIM(x) != 2) {
        PyErr_SetString(PyExc_ValueError, "array must be two-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(x) != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "array must contain floats");
        return NULL;
    }

    N = PyArray_DIM(x, 0);
    D = PyArray_DIM(x, 1);

    if (D > 10) {
        PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
        return NULL;
    }

    /*
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
     */

    data = malloc(N * D * sizeof(double));
    for (i=0; i<N; i++) {
        for (j=0; j<D; j++) {
            double* pd = PyArray_GETPTR2(x, i, j);
            data[i*D + j] = *pd;
        }
    }

    Nleaf = 16;
    treetype = KDTT_DOUBLE;
    //treeoptions = KD_BUILD_SPLIT;
    treeoptions = KD_BUILD_BBOX;

    kd = kdtree_build(NULL, data, N, D, Nleaf,
                      treetype, treeoptions);

    return Py_BuildValue("k", kd);
}

static PyObject* spherematch_kdtree_free(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;

    if (!PyArg_ParseTuple(args, "l", &i)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
        return NULL;
    }

    // Nasty!
    kd = (kdtree_t*)i;

    free(kd->data.any);
    kdtree_free(kd);
    // return None
    return Py_BuildValue("");
}

struct dualtree_results {
    il* inds1;
    il* inds2;
};

static void callback_dualtree(void* v, int ind1, int ind2, double dist2) {
    struct dualtree_results* dtresults = v;
    il_append(dtresults->inds1, ind1);
    il_append(dtresults->inds2, ind2);
}

static PyObject* spherematch_match(PyObject* self, PyObject* args) {
    int i, N;
    long p1, p2;
    kdtree_t *kd1, *kd2;
    double rad;
    struct dualtree_results dtresults;
    PyArrayObject* rtn;
    int dims[2];

    if (!PyArg_ParseTuple(args, "lld", &p1, &p2, &rad)) {
        PyErr_SetString(PyExc_ValueError, "need three args: two kdtree identifiers (ints), and search radius");
        return NULL;
    }

    // Nasty!
    kd1 = (kdtree_t*)p1;
    kd2 = (kdtree_t*)p2;

    dtresults.inds1 = il_new(256);
    dtresults.inds2 = il_new(256);

    dualtree_rangesearch(kd1, kd2, 0.0, rad, NULL,
                         callback_dualtree, &dtresults,
                         NULL, NULL);

    //printf("Found %i close pairs.\n", il_size(dtresults.inds1));

    N = il_size(dtresults.inds1);
    dims[0] = N;
    dims[1] = 2;
    rtn = (PyArrayObject*)PyArray_FromDims(2, dims, PyArray_INT);
    for (i=0; i<N; i++) {
        int* iptr;
        iptr = PyArray_GETPTR2(rtn, i, 0);
        *iptr = kdtree_permute(kd1, il_get(dtresults.inds1, i));
        iptr = PyArray_GETPTR2(rtn, i, 1);
        *iptr = kdtree_permute(kd2, il_get(dtresults.inds2, i));
    }

    il_free(dtresults.inds1);
    il_free(dtresults.inds2);

    // return None
    return PyArray_Return(rtn);
}

static PyMethodDef spherematchMethods[] = {
    { "kdtree_build", spherematch_kdtree_build, METH_VARARGS,
      "build kdtree" },
    { "kdtree_free", spherematch_kdtree_free, METH_VARARGS,
      "free kdtree" },
    { "match", spherematch_match, METH_VARARGS,
      "find matching data points" },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initspherematch_c(void) {
    Py_InitModule("spherematch_c", spherematchMethods);

    import_array();
}

/*
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

 //PyArrayIterObject it;
 //PyArray_ITER_NEXT(it);
 //PyArray_ITER_DATA(it);
}
 */

