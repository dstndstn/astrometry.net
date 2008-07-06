// /opt/local/include/python2.5/
#include "Python.h"

#include <stdio.h>

// PYTHON/site-packages/numpy/core/include
#include "arrayobject.h"
//#include "ufuncobject.h"
//#include "log.h"

static void info(PyArrayObject* x);

static PyObject* spherematch_info(PyObject* self, PyObject* args) {
    int result;
    //const char *command;
    PyArrayObject* x;

    //if (!PyArg_ParseTuple(args, "s", &command))
    //if (!PyArg_ParseTuple(args, .i:function.,&x))
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x))
        return NULL;
    /*
     if (array->nd != 2 || array->descr->type_num != PyArray_DOUBLE) {
     PyErr_SetString(PyExc_ValueError,
     "array must be two-dimensional and of type float");
     */
    info(x);

    result = 0;
    return Py_BuildValue("i", result);
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

