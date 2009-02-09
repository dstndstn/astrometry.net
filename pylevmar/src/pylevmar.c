/* PyLevmar - Python Bindings for Levmar. GPL2.
   Copyright (c) 2006. Alastair Tse <alastair@liquidx.net>
   Copyright (c) 2008. Dustin Lang <dstndstn@gmail.com>
 */

#include "pylevmar.h"

typedef struct _pylm_callback_data {
    PyObject *func;
    PyObject *jacf;
    PyObject *data;
} pylm_callback_data;

void _pylm_callback(PyObject *func, double *p, double *hx, int m, int n, 
                    PyObject *data, int  jacobian) {
    int i;
    PyObject *result = NULL, *args = NULL;
    PyObject *estimate = NULL, *measurement = NULL;
    
    // marshall parameters from c -> python
    estimate = PyTuple_New(m);
    measurement = PyTuple_New(n);
    
    for (i = 0; i < m; i++) {
        PyTuple_SetItem(estimate, i, PyFloat_FromDouble(p[i]));
    }
    for (i = 0; i < n; i++) {
        PyTuple_SetItem(measurement, i, PyFloat_FromDouble(hx[i]));
    }
    
    args = Py_BuildValue("(OOO)", estimate, measurement, data);

    if (!args) {
        goto cleanup;
    }

    // call func
    
    if ((result = PyObject_CallObject(func, args)) == 0) {
        PyErr_Print();
        goto cleanup;
    }

    // marshall results from python -> c
    if ((!jacobian && (PySequence_Size(result) == n)) ||
        (jacobian &&  (PySequence_Size(result) == m*n))) {

        for (i = 0; i < PySequence_Size(result); i++) {
            PyObject *r = PySequence_GetItem(result, i);
            hx[i] = PyFloat_AsDouble(r);
            Py_DECREF(r);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Return value from callback "
                        "should be same size as measurement");
    }

 cleanup:
    
    Py_XDECREF(args);
    Py_XDECREF(estimate);
    Py_XDECREF(measurement);
    Py_XDECREF(result);
    return;
}

void _pylm_func_callback(double *p, double *hx, int m, int n, void *data) {
    pylm_callback_data *pydata = (pylm_callback_data *)data;
    if (pydata->func && PyCallable_Check(pydata->func)) {
        _pylm_callback(pydata->func, p, hx, m, n, pydata->data, 0);
    }
}

void _pylm_jacf_callback(double *p, double *hx, int m, int n, void *data) {
    pylm_callback_data *pydata = (pylm_callback_data *)data;
    if (pydata->jacf && PyCallable_Check(pydata->jacf)) {
        _pylm_callback(pydata->jacf, p, hx, m, n, pydata->data, 1);
    }
}

/* ------- start module methods -------- */

#define PyModule_AddDoubleConstant(mod, name, constant) \
  PyModule_AddObject(mod, name, PyFloat_FromDouble(constant));

void initlevmar() {
    PyObject *mod = Py_InitModule3("levmar", pylm_functions, pylm_doc);
    
    PyModule_AddDoubleConstant(mod, "INIT_MU", LM_INIT_MU);
    PyModule_AddDoubleConstant(mod, "STOP_THRESHU", LM_STOP_THRESH);
    PyModule_AddDoubleConstant(mod, "DIFF_DELTA", LM_DIFF_DELTA);

    PyObject *default_opts = Py_BuildValue("(dddd)", 
                                           LM_INIT_MU, // tau
                                           LM_STOP_THRESH, // eps1
                                           LM_STOP_THRESH, // eps2
                                           LM_STOP_THRESH); //eps3
    PyModule_AddObject(mod, "DEFAULT_OPTS", default_opts);
}

static PyObject *
_pylm_dlevmar_generic(PyObject *mod, PyObject *args, PyObject *kwds,
                     char *argstring, char *kwlist[],
                      int jacobian, int bounds) {
    
    
    PyObject *func = NULL, *jacf = NULL, *initial = NULL, *measurements = NULL;
    PyObject *lower = NULL, *upper = NULL;
    PyObject *opts = NULL, *covar = NULL, *data = NULL;
    PyObject *retval = NULL, *info = NULL;

    pylm_callback_data *pydata = NULL;
    double *c_initial = NULL, *c_measurements = NULL, *c_opts = NULL;
    double *c_lower = NULL, *c_upper = NULL;
    int max_iter = 0, run_iter = 0, i = 0, m = 0, n = 0;
    double c_info[LM_INFO_SZ];

    // parse arguments
    if (!bounds) {
        if (jacobian) {
            if (!PyArg_ParseTupleAndKeywords(args, kwds, argstring, kwlist,
                                             &func, &jacf, &initial,
                                             &measurements, &max_iter, 
                                             &opts, &covar, &data))
                return NULL;
        } else {
            if (!PyArg_ParseTupleAndKeywords(args, kwds, argstring, kwlist,
                                             &func, &initial,
                                             &measurements, &max_iter, 
                                             &opts, &covar, &data))
                return NULL;
        }
    } else {
        if (jacobian) {
            if (!PyArg_ParseTupleAndKeywords(args, kwds, argstring, kwlist,
                                             &func, &jacf, &initial,
                                             &measurements, &lower, &upper, &max_iter, 
                                             &opts, &covar, &data))
                return NULL;
        } else {
            if (!PyArg_ParseTupleAndKeywords(args, kwds, argstring, kwlist,
                                             &func, &initial,
                                             &measurements, &lower, &upper, &max_iter,
                                             &opts, &covar, &data))
                return NULL;
        }
    }
     
    // check each var type
   if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "func must be a callable object");
        return NULL;
    }

    if (!PySequence_Check(initial)) {
        PyErr_SetString(PyExc_TypeError, "initial must be a sequence type");
        return NULL;
    }

    if (!PySequence_Check(measurements)) {
        PyErr_SetString(PyExc_TypeError, "measurements must be a sequence type");
        return NULL;
    }

    if (jacobian && !PyCallable_Check(jacf)) {
        PyErr_SetString(PyExc_TypeError, "jacf must be a callable object");
        return NULL;
    }        

    if (lower && !PySequence_Check(lower)) {
        PyErr_SetString(PyExc_TypeError, "lower bounds must be a sequence type");
        return NULL;
    }
    if (upper && !PySequence_Check(upper)) {
        PyErr_SetString(PyExc_TypeError, "upper bounds must be a sequence type");
        return NULL;
    }

    if (opts && !PySequence_Check(opts) && (PySequence_Size(opts) < 4)) {
        PyErr_SetString(PyExc_TypeError, "opts must be a sequence/tuple "
                        "of length 4.");
        return NULL;
    }

    // convert python types into C

    pydata = PyMem_Malloc(sizeof(pydata));

    pydata->func = func;
    pydata->jacf = jacf;
    if (!data)
        pydata->data = Py_None;
    else
        pydata->data = data;

    Py_XINCREF(pydata->data);

    m = PySequence_Size(initial);
    n = PySequence_Size(measurements);

    c_initial = PyMem_Malloc(sizeof(double) * m);
    c_measurements = PyMem_Malloc(sizeof(double) * n); 

    if (lower)
        c_lower = PyMem_Malloc(sizeof(double) * m);
    if (upper)
        c_upper = PyMem_Malloc(sizeof(double) * m);

    if (!pydata || !c_initial || !c_measurements ||
        (lower && !c_lower) || (upper && !c_upper)) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory");
        return NULL;
    }

    for (i = 0; i < m; i++) {
        PyObject *r = PySequence_GetItem(initial, i);
        c_initial[i] = PyFloat_AsDouble(r);
        Py_XDECREF(r);
    }

    if (lower)
        for (i = 0; i < m; i++) {
            PyObject *r = PySequence_GetItem(lower, i);
            c_lower[i] = PyFloat_AsDouble(r);
            Py_XDECREF(r);
        }
    if (upper)
        for (i = 0; i < m; i++) {
            PyObject *r = PySequence_GetItem(upper, i);
            c_upper[i] = PyFloat_AsDouble(r);
            Py_XDECREF(r);
        }

    for (i = 0; i < n; i++) {
        PyObject *r = PySequence_GetItem(measurements, i);
        c_measurements[i] = PyFloat_AsDouble(r);
        Py_XDECREF(r);
    }

    if (opts) {
        c_opts = PyMem_Malloc(sizeof(double) * 4);
        for (i = 0; i < 4; i++) {
            PyObject *r = PySequence_GetItem(opts, i);
            c_opts[i] = PyFloat_AsDouble(r);
            Py_XDECREF(r);
        }
    }
    
    // call func
    if (!bounds) {
        if (jacobian) {
            run_iter =  dlevmar_der(_pylm_func_callback, _pylm_jacf_callback,
                                    c_initial, c_measurements, m, n,
                                    max_iter, c_opts, c_info, NULL, NULL, pydata);
        } else {
            run_iter =  dlevmar_dif(_pylm_func_callback, c_initial, c_measurements,
                                    m, n, max_iter, c_opts, c_info, NULL, NULL, pydata);
        }
    } else {
        if (jacobian) {
            run_iter =  dlevmar_bc_der(_pylm_func_callback, _pylm_jacf_callback,
                                       c_initial, c_measurements, m, n,
                                       c_lower, c_upper,
                                       max_iter, c_opts, c_info, NULL, NULL, pydata);
        } else {
            run_iter =  dlevmar_bc_dif(_pylm_func_callback, c_initial, c_measurements,
                                       m, n, c_lower, c_upper,
                                       max_iter, c_opts, c_info, NULL, NULL, pydata);
        }
    }

    // convert results back into python
    if (run_iter > 0) {
        retval = PyTuple_New(m);
        for (i = 0; i < m; i++) {
            PyTuple_SetItem(retval, i, PyFloat_FromDouble(c_initial[i]));
        }
    }
    else {
        retval = Py_None;
        Py_INCREF(Py_None);
    }

    // convert additional information into python
    info = Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
                         "initial_e2", c_info[0],
                         "estimate_e2", c_info[1],
                         "estimate_Jt", c_info[2],
                         "estimate_Dp2", c_info[3],
                         "estimate_mu", c_info[4],
                         "iterations", c_info[5],
                         "termination", c_info[6],
                         "function_evaluations", c_info[7],
                         "jacobian_evaluations", c_info[8]);

    if (c_measurements) {
        PyMem_Free(c_measurements); c_measurements = NULL;
    }
    if (c_initial) {
        PyMem_Free(c_initial); c_initial = NULL;
    }
    if (c_opts) {
        PyMem_Free(c_opts); c_opts = NULL;
    }
    if (c_lower) {
        PyMem_Free(c_lower); c_lower = NULL;
    }
    if (c_upper) {
        PyMem_Free(c_upper); c_upper = NULL;
    }
    if (pydata) {
        Py_XDECREF(pydata->data);
        PyMem_Free(pydata); pydata = NULL;
    }

    return Py_BuildValue("(OiO)", retval, run_iter, info, NULL);
}

static PyObject *
pylm_dlevmar_der(PyObject *mod, PyObject *args, PyObject *kwds)
{
    char *argstring = "OOOOi|OOO";
    char *kwlist[] = {"func", "jacf", "initial", "measurements",
                      "max_iter", "opts", "covar", "data", 
                      NULL};
    return _pylm_dlevmar_generic(mod, args, kwds, argstring, kwlist, 1, 0);
}

static PyObject *
pylm_dlevmar_dif(PyObject *mod, PyObject *args, PyObject *kwds)
{
    char *argstring = "OOOi|OOO";
    char *kwlist[] = {"func", "initial", "measurements", "max_iter",
                      "opts", "covar", "data", NULL};
    return _pylm_dlevmar_generic(mod, args, kwds, argstring, kwlist, 0, 0);
}

static PyObject *
pylm_dlevmar_bc_der(PyObject *mod, PyObject *args, PyObject *kwds)
{
    char *argstring = "OOOOOOi|OOO";
    char *kwlist[] = {"func", "jacf", "initial", "measurements",
                      "lower", "upper",
                      "max_iter", "opts", "covar", "data", 
                      NULL};
    return _pylm_dlevmar_generic(mod, args, kwds, argstring, kwlist, 1, 1);
}

static PyObject *
pylm_dlevmar_bc_dif(PyObject *mod, PyObject *args, PyObject *kwds)
{
    char *argstring = "OOOOOi|OOO";
    char *kwlist[] = {"func", "initial", "measurements",
                      "lower", "upper",
                      "max_iter", "opts", "covar", "data", NULL};
    return _pylm_dlevmar_generic(mod, args, kwds, argstring, kwlist, 0, 1);
}

static PyObject *
pylm_dlevmar_chkjac(PyObject *mod, PyObject *args, PyObject *kwds)
{
    PyObject *func = NULL, *jacf = NULL, *initial = NULL;
    PyObject *data = NULL;
    PyObject *retval = NULL;

    pylm_callback_data *pydata = NULL;
    double *c_initial = NULL, *err = NULL;
    int i = 0, m = 0, n = 0;


    static char *kwlist[] = {"func", "jacf", "initial", "n", "data", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOi|O", kwlist,
                                     &func, &jacf, &initial, &n, &data))
        return NULL;
                                     
    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "func must be a callable object");
        return NULL;
    }

    if (!PyCallable_Check(jacf)) {
        PyErr_SetString(PyExc_TypeError, "jacf must be a callable object");
        return NULL;
    }

    if (!PySequence_Check(initial)) {
        PyErr_SetString(PyExc_TypeError, "initial must be a sequence type");
        return NULL;
    }

    // convert python types into C
    m = PySequence_Size(initial);
    pydata = PyMem_Malloc(sizeof(pydata));
    c_initial = PyMem_Malloc(sizeof(double) * m);
    err = PyMem_Malloc(sizeof(double) * n);

    if (!pydata || !c_initial) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory");
        return NULL;
    }

    pydata->func = func;
    pydata->jacf = jacf;
    pydata->data = data;
    Py_INCREF(data);

    for (i = 0; i < m; i++) {
        PyObject *r = PySequence_GetItem(initial, i);
        c_initial[i] = PyFloat_AsDouble(r);
        Py_XDECREF(r);
    }

    // call func
    dlevmar_chkjac(_pylm_func_callback, _pylm_jacf_callback, 
                   c_initial, m, n, pydata, err);

    // convert results back into python
    retval = PyTuple_New(n);
    for (i = 0; i < n; i++) {
        PyTuple_SetItem(retval, i, PyFloat_FromDouble(err[i]));
    }

    if (c_initial) {
        PyMem_Free(c_initial); c_initial = NULL;
    }

    if (pydata) {
        PyMem_Free(pydata); pydata = NULL;
    }

    Py_XDECREF(data);
    return retval;
}
