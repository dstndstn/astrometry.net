#include <stdio.h>

// /opt/local/include/python2.5/
#include "Python.h"
// PYTHON/site-packages/numpy/core/include
#include "arrayobject.h"
//#include "ufuncobject.h"
//#include "log.h"

/*
#include <Python.h>
PyObject *wrap_function(PyObject *self, PyObject *args) {
 int x, result;
 if (!PyArg_ParseTuple(args, .i:function.,&x))
   return NULL;
  result = function(x);
 return Py_BuildValue(.i.,result);
}

Static PyMethod exampleMethods[] = {
  {.function., wrap_function, 1},
  {NULL, NULL}
};
void initialize_function(){
  PyObject *m;
  m = Py_InitModule(.example., .exampleMethods.);
}
 */

void spherematch_info(PyArrayObject* x) {
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

