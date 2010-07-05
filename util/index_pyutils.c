#include "Python.h"

// numpy - this should be in site-packages/numpy/core/include
#include "arrayobject.h"

#include "starkd.h"
#include "codekd.h"
#include "starutil.h"
#include "qidxfile.h"
#include "quadfile.h"

static PyObject* quadfile_get_stars_for_quads(PyObject* self, PyObject* args) {
	PyArrayObject* pyquads;
	PyArrayObject* pystars;
	int N, DQ;
	quadfile* qf;
	npy_intp dims[2];
	int i;

    if (!PyArg_ParseTuple(args, "lO!", &qf, &PyArray_Type, &pyquads)) {
        PyErr_SetString(PyExc_ValueError, "need: quadfile and quads (numpy array)");
        return NULL;
	}

    N = PyArray_DIM(pyquads, 0);
	dims[0] = N;
	DQ = quadfile_dimquads(qf);
	dims[1] = DQ;
	pystars = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_UINT);
	//printf("N %i, DQ %i\n", N, DQ);

	for (i=0; i<N; i++) {
		int quad = *(int*)(PyArray_GETPTR1(pyquads, i));
		//printf("quad %i\n", quad);
		if (quadfile_get_stars(qf, quad, (unsigned int*)PyArray_GETPTR1(pystars, i))) {
			PyErr_SetString(PyExc_ValueError, "quadfile_get_stars() failed");
			return NULL;
		}
	}
	return Py_BuildValue("O", pystars);
}

static PyObject* qidxfile_get_quad_list(PyObject* self, PyObject* args) {
    PyArrayObject* pyquads;
	uint32_t* quads;
	int nquads;
	qidxfile* qf;
	int starid;
	npy_intp dims[2];

    if (!PyArg_ParseTuple(args, "ll", &qf, &starid)) {
        PyErr_SetString(PyExc_ValueError, "need: qidxfile and starid");
        return NULL;
	}

	if (qidxfile_get_quads(qf, starid, &quads, &nquads)) {
		PyErr_SetString(PyExc_ValueError, "qidxfile_get_quads() failed.");
        return NULL;
	}

	dims[0] = nquads;
	pyquads = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_INT, quads);
    return Py_BuildValue("O", pyquads);
}

// starkd_search_stars_numpy(addr, ra, dec, radius)
static PyObject* starkd_search_stars_numpy(PyObject* self, PyObject* args) {
	startree_t* s;
	double ra, dec, radius;
    int N;
    PyArrayObject* pyxyz;
    PyArrayObject* pyradec;
    PyArrayObject* pyinds;
	npy_intp dims[2];
	double xyz[3];
	double r2;
	double* xyzres;
	double* radecres;
	int* inds;

    if (!PyArg_ParseTuple(args, "lddd", &s, &ra, &dec, &radius)) {
        PyErr_SetString(PyExc_ValueError, "need four args: starkd, ra, dec, radius");
        return NULL;
	}

	radecdeg2xyzarr(ra, dec, xyz);
	r2 = deg2distsq(radius);

	startree_search_for(s, xyz, r2, &xyzres, &radecres, &inds, &N);

	dims[0] = N;
	dims[1] = 3;
	pyxyz = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, xyzres);

	dims[0] = N;
	dims[1] = 2;
	pyradec = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, radecres);

	dims[0] = N;
	dims[1] = 1;
	pyinds = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, PyArray_INT, inds);

    return Py_BuildValue("(OOO)", pyxyz, pyradec, pyinds);
}


static PyObject* codekd_get_codes_numpy(PyObject* self, PyObject* args) {
    int N, D;
    int i;
    PyArrayObject* A;
	codetree* t;
	npy_intp dims[2];

    if (!PyArg_ParseTuple(args, "l", &t)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: codetree pointer (int)");
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

static PyObject* starkd_get_stars_numpy(PyObject* self, PyObject* args) {
    int N, D;
    int i;
    PyArrayObject* A;
	startree_t* t;
	npy_intp dims[2];

    if (!PyArg_ParseTuple(args, "l", &t)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: startree pointer (int)");
        return NULL;
	}

	N = startree_N(t);
	D = startree_D(t);
	dims[0] = N;
	dims[1] = D;

	A = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
	for (i=0; i<startree_N(t); i++)
		if (startree_get(t, i, PyArray_GETPTR1(A, i))) {
			PyErr_SetString(PyExc_ValueError, "failed to retrieve star");
			return NULL;
		}
    return Py_BuildValue("O", A);
}



static PyMethodDef myMethods[] = {
    { "codekd_get_codes_numpy", codekd_get_codes_numpy, METH_VARARGS,
      "Get all codes as a numpy array." },
    { "starkd_get_stars_numpy", starkd_get_stars_numpy, METH_VARARGS,
      "Get all stars as a numpy array." },
    { "starkd_search_stars_numpy", starkd_search_stars_numpy, METH_VARARGS,
      "Search for stars in a region; return numpy arrays." },
    { "qidxfile_get_quad_list", qidxfile_get_quad_list, METH_VARARGS,
      "Finds the quads using a given star; returns numpy array." },
	{ "quadfile_get_stars_for_quads", quadfile_get_stars_for_quads, METH_VARARGS,
	  "Returns the stars that are part of the given set of quads." },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_index_util(void) {
    Py_InitModule("_index_util", myMethods);
    import_array();
}
