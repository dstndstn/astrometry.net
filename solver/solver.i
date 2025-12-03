/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */
%module(package="astrometry.solver") solver

%include <typemaps.i>
%include <cstring.i>
%include <exception.i>

%{
// numpy.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>

#include "os-features.h"
#include "verify.h"
%}

%init %{
    // start solver.i %init block
    // init numpy
#if SWIG_VERSION >= 0x040400
    import_array1(-1);
#else
    import_array();
#endif
    // end solver.i %init block
%}

%import "../util/util.i"

%include "verify.h"

%inline %{

static PyObject* verify_star_lists_np(PyObject* pyrefxy,
                                   PyObject* pytestxy,
                                   PyObject* pytestsig2,
                                   double effective_area,
                                   double distractors,
                                   double logodds_bail,
                                   double logodds_accept) {
    PyArrayObject *np_refxy, *np_testxy, *np_testsig2;
    PyArray_Descr* dtype = NULL;
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
        NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
    double* refxy, *testxy, *testsig2;
    int NT, NR;
    int two;
    int n;
    double logodds;

    dtype = PyArray_DescrFromType(NPY_DOUBLE);

    Py_INCREF(dtype);
    np_refxy = (PyArrayObject*)PyArray_FromAny(pyrefxy, dtype, 2, 2, req, NULL);
    if (!np_refxy) {
        PyErr_SetString(PyExc_ValueError,"Expected refxy array to be double");
        Py_DECREF(dtype);
        return NULL;
    }
    NR = PyArray_DIM(np_refxy, 0);
    two = PyArray_DIM(np_refxy, 1);
    if (two != 2) {
        PyErr_SetString(PyExc_ValueError,"Expected refxy array to be size 2xNR");
        Py_DECREF(np_refxy);
        return NULL;
    }

    Py_INCREF(dtype);
    np_testxy = (PyArrayObject*)PyArray_FromAny(pytestxy, dtype, 2, 2, req, NULL);
    if (!np_testxy) {
        PyErr_SetString(PyExc_ValueError,"Expected testxy array to be double");
        Py_DECREF(dtype);
        return NULL;
    }
    NT = PyArray_DIM(np_testxy, 0);
    two = PyArray_DIM(np_testxy, 1);
    if (two != 2) {
        PyErr_SetString(PyExc_ValueError,"Expected testxy array to be size 2xNR");
        Py_DECREF(np_testxy);
        return NULL;
    }

    Py_INCREF(dtype);
    np_testsig2 = (PyArrayObject*)PyArray_FromAny(pytestsig2, dtype, 1, 1, req, NULL);
    if (!np_testsig2) {
        PyErr_SetString(PyExc_ValueError,"Expected testsig2 array to be double");
        Py_DECREF(dtype);
        return NULL;
    }
    n = PyArray_DIM(np_testsig2, 0);
    if (n != NT) {
        PyErr_SetString(PyExc_ValueError,"Expected testsig2 array to be size NT");
        Py_DECREF(np_testsig2);
        return NULL;
    }
    Py_CLEAR(dtype);

    refxy = PyArray_DATA(np_refxy);
    testxy = PyArray_DATA(np_testxy);
    testsig2 = PyArray_DATA(np_testsig2);

    logodds = verify_star_lists(refxy, NR, testxy, testsig2, NT,
                                effective_area, distractors, logodds_bail,
                                logodds_accept, NULL, NULL, NULL, NULL, NULL);

    Py_DECREF(np_refxy);
    Py_DECREF(np_testxy);
    Py_DECREF(np_testsig2);

    return PyFloat_FromDouble(logodds);
 }

 %}
