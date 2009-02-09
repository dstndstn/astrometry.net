/* PyLevmar - Python Bindings for Levmar. GPL2.
   Copyright (c) 2006. Alastair Tse <alastair@liquidx.net>
   Copyright (c) 2008. Dustin Lang <dstndstn@gmail.com>
*/

#ifndef __PYLEVMAR_H
#define __PYLEVMAR_H

#include "lm.h"

#include <Python.h>
#include "structmember.h"

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

static PyObject *
pylm_dlevmar_der(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_dlevmar_dif(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_dlevmar_bc_der(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_dlevmar_bc_dif(PyObject *mod, PyObject *args, PyObject *kwds);
/*
static PyObject *
pylm_dlevmar_lec_der(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_dlevmar_lec_dif(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_slevmar_der(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_slevmar_dif(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_slevmar_bc_der(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_slevmar_bc_dif(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_slevmar_lec_der(PyObject *mod, PyObject *args, PyObject *kwds);
static PyObject *
pylm_slevmar_lec_dif(PyObject *mod, PyObject *args, PyObject *kwds);

static PyObject *
pylm_slevmar_chkjac(PyObject *mod, PyObject *args, PyObject *kwds);
*/

static PyObject *
pylm_dlevmar_chkjac(PyObject *mod, PyObject *args, PyObject *kwds);


static char *pylm_doc = \
"Python Bindings to Levmar Non-Linear Regression Solver.\n\n" 
"Members:\n\n"
"  levmar.DEFAULT_OPTS = 4 element tuple representing the default\n"
"                        opts into dder an ddif.\n"
"  levmar.INIT_MU      = Initial value for mu\n"
"  levmar.STOP_THRESH  = Stopping threshold\n"
"  levmar.DIFF_DELTA   = Differential delta\n";


static PyMethodDef pylm_functions[] = {
    {
        "dder",
        (PyCFunction)pylm_dlevmar_der,
        METH_VARARGS|METH_KEYWORDS,
        "dlevmar_der(func, jacf, estimate, measurements, itmax,\n"
        "            opts = None, covar = None, data = None)\n"
        "-> returns: (result, iterations, run_info)\n"
        "\n"
    },
    {
        "ddif",
        (PyCFunction)pylm_dlevmar_dif,
        METH_VARARGS|METH_KEYWORDS,
        "dlevmar_der(func, estimate, measurements, itmax,\n"
        "            opts = None, covar = None, data = None)\n"
        "-> returns: (result, iterations, run_info)\n"
        "\n"
    },
    {
        "dchkjac",
        (PyCFunction)pylm_dlevmar_chkjac,
        METH_VARARGS|METH_KEYWORDS,
        "dlevmar_der(func, jacf, initial, m, data = None) -> tuple\n\n"
        "Check a function and the jacobian and the relative error at point\n\n"
        "initial."
    },
    {
        "ddif_bc",
        (PyCFunction)pylm_dlevmar_bc_dif,
        METH_VARARGS|METH_KEYWORDS,
        "dlevmar_bc_dif(func, estimate, measurements, lower, upper, itmax,\n"
        "               opts = None, covar = None, data = None)\n"
        "-> returns: (result, iterations, run_info)\n"
        "\n"
    },
    {
        "dder_bc",
        (PyCFunction)pylm_dlevmar_bc_der,
        METH_VARARGS|METH_KEYWORDS,
        "dlevmar_bc_der(func, estimate, measurements, lower, upper, itmax,\n"
        "               opts = None, covar = None, data = None)\n"
        "-> returns: (result, iterations, run_info)\n"
        "\n"
    },
    {NULL, NULL, 0, NULL}
};

#endif
