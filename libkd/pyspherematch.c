/*
  This file is part of libkd.
  Copyright 2008, 2009, 2010, 2011 Dustin Lang.

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
#include <assert.h>

// numpy - this should be in site-packages/numpy/core/include
#include "arrayobject.h"

#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "dualtree_rangesearch.h"
#include "dualtree_nearestneighbour.h"
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
        PyErr_SetString(PyExc_ValueError, "array must contain doubles");
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

static PyObject* spherematch_kdtree_open(PyObject* self, PyObject* args) {
    kdtree_t* kd;
    char* fn;

    if (!PyArg_ParseTuple(args, "s", &fn)) {
        PyErr_SetString(PyExc_ValueError, "need one args: kdtree filename");
        return NULL;
    }

    kd = kdtree_fits_read(fn, NULL, NULL);
    return Py_BuildValue("k", kd);
}

static PyObject* spherematch_kdtree_close(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;

    if (!PyArg_ParseTuple(args, "l", &i)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;
    kdtree_fits_close(kd);
    return Py_BuildValue("");
}

static PyObject* spherematch_kdtree_n(PyObject* self, PyObject* args) {
    long i;
    kdtree_t* kd;
    if (!PyArg_ParseTuple(args, "l", &i)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
        return NULL;
    }
    // Nasty!
    kd = (kdtree_t*)i;
    return PyInt_FromLong(kdtree_n(kd));
}

struct dualtree_results {
    il* inds1;
    il* inds2;
    dl* dists;
};

static void callback_dualtree(void* v, int ind1, int ind2, double dist2) {
    struct dualtree_results* dtresults = v;
    il_append(dtresults->inds1, ind1);
    il_append(dtresults->inds2, ind2);
    dl_append(dtresults->dists, sqrt(dist2));
}

static PyObject* spherematch_match(PyObject* self, PyObject* args) {
    int i, N;
    long p1, p2;
    kdtree_t *kd1, *kd2;
    double rad;
    struct dualtree_results dtresults;
    PyArrayObject* inds;
    npy_intp dims[2];
    PyArrayObject* dists;
	anbool notself;
	PyObject* rtn;
	
	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "lldb", &p1, &p2, &rad, &notself)) {
        PyErr_SetString(PyExc_ValueError, "need three args: two kdtree identifiers (ints), and search radius");
        return NULL;
    }
	//printf("Notself = %i\n", (int)notself);
    // Nasty!
    kd1 = (kdtree_t*)p1;
    kd2 = (kdtree_t*)p2;

    dtresults.inds1 = il_new(256);
    dtresults.inds2 = il_new(256);
    dtresults.dists = dl_new(256);
    dualtree_rangesearch(kd1, kd2, 0.0, rad, notself, NULL,
                         callback_dualtree, &dtresults,
                         NULL, NULL);

    N = il_size(dtresults.inds1);
    dims[0] = N;
    dims[1] = 2;

    inds =  (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_INT);
    dims[1] = 1;
    dists = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
    for (i=0; i<N; i++) {
        int* iptr;
        double* dptr;
        iptr = PyArray_GETPTR2(inds, i, 0);
        *iptr = kdtree_permute(kd1, il_get(dtresults.inds1, i));
        iptr = PyArray_GETPTR2(inds, i, 1);
        *iptr = kdtree_permute(kd2, il_get(dtresults.inds2, i));
        dptr = PyArray_GETPTR2(dists, i, 0);
        *dptr = dl_get(dtresults.dists, i);
    }

    il_free(dtresults.inds1);
    il_free(dtresults.inds2);
    dl_free(dtresults.dists);

    rtn = Py_BuildValue("(OO)", inds, dists);
    Py_DECREF(inds);
    Py_DECREF(dists);
    return rtn;
}

static PyObject* spherematch_nn(PyObject* self, PyObject* args) {
    int i, NY;
    long p1, p2;
    kdtree_t *kd1, *kd2;
    npy_intp dims[1];
    PyArrayObject* inds;
    PyArrayObject* dist2s;
    int *pinds;
    double *pdist2s;
    double rad;
	anbool notself;
	int* tempinds;
	PyObject* rtn;

	// So that ParseTuple("b") with a C "anbool" works
	assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "lldb", &p1, &p2, &rad, &notself)) {
        PyErr_SetString(PyExc_ValueError, "need three args: two kdtree identifiers (ints), and search radius");
        return NULL;
    }
    // Nasty!
    kd1 = (kdtree_t*)p1;
    kd2 = (kdtree_t*)p2;

    NY = kdtree_n(kd2);

    dims[0] = NY;
    inds   = (PyArrayObject*)PyArray_SimpleNew(1, dims, PyArray_INT);
    dist2s = (PyArrayObject*)PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
    assert(PyArray_ITEMSIZE(inds) == sizeof(int));
    assert(PyArray_ITEMSIZE(dist2s) == sizeof(double));

	// YUCK!
	tempinds = (int*)malloc(NY * sizeof(int));
	double* tempdists = (double*)malloc(NY * sizeof(double));

    pinds   = tempinds; //PyArray_DATA(inds);
    //pdist2s = PyArray_DATA(dist2s);
	pdist2s = tempdists;

    dualtree_nearestneighbour(kd1, kd2, rad*rad, &pdist2s, &pinds, notself);

	// now we have to apply kd1's permutation array!
	for (i=0; i<NY; i++)
		if (pinds[i] != -1)
			pinds[i] = kdtree_permute(kd1, pinds[i]);


	pinds = PyArray_DATA(inds);
    pdist2s = PyArray_DATA(dist2s);

	for (i=0; i<NY; i++) {
		pinds[i] = -1;
		pdist2s[i] = HUGE_VAL;
	}
	// and apply kd2's permutation array!
	for (i=0; i<NY; i++) {
		if (tempinds[i] != -1) {
			int j = kdtree_permute(kd2, i);
			pinds[j] = tempinds[i];
			pdist2s[j] = tempdists[i];
		}
	}
	free(tempinds);
	free(tempdists);

	rtn = Py_BuildValue("(OO)", inds, dist2s);
	Py_DECREF(inds);
	Py_DECREF(dist2s);
	return rtn;
}

static PyObject* spherematch_kdtree_bbox(PyObject* self, PyObject* args) {
  PyArrayObject* bbox;
  PyObject* rtn;
  npy_intp dims[2];
  long i;
  double *bb;
  anbool ok;
  kdtree_t* kd;
  int j, D;

  if (!PyArg_ParseTuple(args, "l", &i)) {
    PyErr_SetString(PyExc_ValueError, "need one arg: kdtree identifier (int)");
    return NULL;
  }
  // Nasty!
  kd = (kdtree_t*)i;
  D = kd->ndim;
  dims[0] = D;
  dims[1] = 2;
  bbox = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  {
    double bblo[D];
    double bbhi[D];
    ok = kdtree_get_bboxes(kd, 0, bblo, bbhi);
    assert(ok);
    bb = PyArray_DATA(bbox);
    for (j=0; j<D; j++) {
      bb[j*2 + 0] = bblo[j];
      bb[j*2 + 1] = bbhi[j];
    }
  }
  rtn = Py_BuildValue("O", bbox);
  Py_DECREF(bbox);
  return rtn;
}


static PyObject* spherematch_nn2(PyObject* self, PyObject* args) {
  int i, j, NY, N;
  long p1, p2;
  kdtree_t *kd1, *kd2;
  npy_intp dims[1];
  PyArrayObject* I;
  PyArrayObject* J;
  PyArrayObject* dist2s;
  int *pi;
  int *pj;
  double *pd;
  double rad;
  anbool notself;
  int* tempinds;
  double* tempd2;
  PyObject* rtn;

  // So that ParseTuple("b") with a C "anbool" works
  assert(sizeof(anbool) == sizeof(unsigned char));

  if (!PyArg_ParseTuple(args, "lldb", &p1, &p2, &rad, &notself)) {
    PyErr_SetString(PyExc_ValueError, "need three args: two kdtree identifiers (ints), and search radius");
    return NULL;
  }
  // Nasty!
  kd1 = (kdtree_t*)p1;
  kd2 = (kdtree_t*)p2;

  // quick check for no-overlap case
  if (kdtree_node_node_mindist2_exceeds(kd1, 0, kd2, 0, rad*rad)) {
    // allocate empty return arrays
    dims[0] = 0;
    I = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    J = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    dist2s = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    rtn = Py_BuildValue("(OOO)", I, J, dist2s);
    Py_DECREF(I);
    Py_DECREF(J);
    Py_DECREF(dist2s);
    return rtn;
  }

  NY = kdtree_n(kd2);

  tempinds = (int*)malloc(NY * sizeof(int));
  tempd2 = (double*)malloc(NY * sizeof(double));

  dualtree_nearestneighbour(kd1, kd2, rad*rad, &tempd2, &tempinds, notself);

  // count number of matches
  N = 0;
  for (i=0; i<NY; i++)
    if (tempinds[i] != -1)
      N++;

  // allocate return arrays
  dims[0] = N;
  I = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
  J = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
  dist2s = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  pi = PyArray_DATA(I);
  pj = PyArray_DATA(J);
  pd = PyArray_DATA(dist2s);

  j = 0;
  for (i=0; i<NY; i++) {
    if (tempinds[i] == -1)
      continue;
    pi[j] = kdtree_permute(kd1, tempinds[i]);
    pj[j] = kdtree_permute(kd2, i);
    pd[j] = tempd2[i];
    j++;
  }

  free(tempinds);
  free(tempd2);

  rtn = Py_BuildValue("(OOO)", I, J, dist2s);
  Py_DECREF(I);
  Py_DECREF(J);
  Py_DECREF(dist2s);
  return rtn;
}




static PyMethodDef spherematchMethods[] = {
    { "kdtree_build", spherematch_kdtree_build, METH_VARARGS,
      "build kdtree" },
    { "kdtree_write", spherematch_kdtree_write, METH_VARARGS,
      "save kdtree to file" },
    { "kdtree_open", spherematch_kdtree_open, METH_VARARGS,
      "open kdtree from file" },
    { "kdtree_close", spherematch_kdtree_close, METH_VARARGS,
      "close kdtree opened with kdtree_open" },
    { "kdtree_free", spherematch_kdtree_free, METH_VARARGS,
      "free kdtree" },

    { "kdtree_bbox", spherematch_kdtree_bbox, METH_VARARGS,
      "get bounding-box of this tree" },
    { "kdtree_n", spherematch_kdtree_n, METH_VARARGS,
      "N pts in tree" },

    { "match", spherematch_match, METH_VARARGS,
      "find matching data points" },
    { "nearest", spherematch_nn, METH_VARARGS,
      "find nearest neighbours" },

    { "nearest2", spherematch_nn2, METH_VARARGS,
      "find nearest neighbours (different return values)" },

    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initspherematch_c(void) {
    Py_InitModule("spherematch_c", spherematchMethods);
    import_array();
}

