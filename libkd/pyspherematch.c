/*
  This file is part of libkd.
  Copyright 2008 Dustin Lang.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "Python.h"

#include <stdio.h>

// numpy - this should be in site-packages/numpy/core/include
#include "arrayobject.h"

#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "dualtree_rangesearch.h"
#include "bl.h"

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
    return Py_BuildValue("");
}

static PyObject* spherematch_kdtree_write(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;
    char* fn;
    int rtn;

    if (!PyArg_ParseTuple(args, "ls", &i, &fn)) {
        PyErr_SetString(PyExc_ValueError, "need two args: kdtree identifier (int), filename (string)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;

    rtn = kdtree_fits_write(kd, fn, NULL);
    return Py_BuildValue("i", rtn);
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

    return PyArray_Return(rtn);
}

static PyMethodDef spherematchMethods[] = {
    { "kdtree_build", spherematch_kdtree_build, METH_VARARGS,
      "build kdtree" },
    { "kdtree_write", spherematch_kdtree_write, METH_VARARGS,
      "save kdtree to file" },
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

