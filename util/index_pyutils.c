#include "Python.h"

// numpy - this should be in site-packages/numpy/core/include
#include "arrayobject.h"

#include "starkd.h"
#include "codekd.h"



static PyObject* codekd_get_codes_numpy(PyObject* self, PyObject* args) {
    int N, D;
    int i,j;
    //kdtree_t* kd;
    //double* data;
    PyArrayObject* A;
	codetree* t;
	npy_intp dims[2];

    if (!PyArg_ParseTuple(args, "l", &t)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: codetree identifier (int)");
        return NULL;
	}

	N = codetree_N(t);
	D = codetree_D(t);
	dims[0] = N;
	dims[1] = D;

	A = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
	for (i=0; i<codetree_N(t); i++)
		if (codetree_get(t, i, PyArray_GETPTR1(A, i))) {
			PyErr_SetString(PyExc_ValueError, "failed to retrieve code");
			return NULL;
		}
    return Py_BuildValue("O", A);
}


static PyMethodDef myMethods[] = {
    { "codekd_get_codes_numpy", codekd_get_codes_numpy, METH_VARARGS,
      "Get all codes as a numpy array." },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_index_util(void) {
    Py_InitModule("_index_util", myMethods);
    import_array();
}
