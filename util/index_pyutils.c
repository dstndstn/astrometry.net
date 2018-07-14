/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
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

static PyObject* pyval_int(void* v, int index) {
    PyObject* pyval;
    int* i = (int*)v;
    pyval = PyInt_FromLong((long)i[index]);
    assert(pyval);
    return pyval;
}
static PyObject* pyval_int64(void* v, int index) {
    PyObject* pyval;
    int64_t* i = (int64_t*)v;
    pyval = PyLong_FromLongLong((long long)i[index]);
    assert(pyval);
    return pyval;
}
static PyObject* pyval_double(void* v, int index) {
    PyObject* pyval;
    double* i = (double*)v;
    pyval = PyFloat_FromDouble(i[index]);
    assert(pyval);
    return pyval;
}

static PyObject* array_to_pylist(void* X, int N, PyObject* (*pyval)(void* v, int i)) {
    PyObject* pylist;
    int i;
    pylist = PyList_New(N);
    assert(pylist);
    for (i=0; i<N; i++) {
        PyObject* v;
        v = pyval(X, i);
        assert(v);
        PyList_SetItem(pylist, i, v);
    }
    return pylist;
}
static PyObject* array_to_pylist2(void* vdata, int N, int D, PyObject* (*pyval)(void* v, int i)) {
    PyObject* pylist;
    int i,j;
    pylist = PyList_New(N);
    assert(pylist);
    for (i=0; i<N; i++) {
        PyObject* v;
        PyObject* row = PyList_New(D);
        assert(row);
        for (j=0; j<D; j++) {
            v = pyval(vdata, i*D+j);
            assert(v);
            PyList_SetItem(row, j, v);
        }
        PyList_SetItem(pylist, i, row);
    }
    return pylist;
}

static PyObject* arrayd_to_pylist2(double* xyz, int N, int D) {
    return array_to_pylist2(xyz, N, D, pyval_double);
}
static PyObject* arrayi_to_pylist(int* X, int N) {
    return array_to_pylist(X, N, pyval_int);
}


// starkd_search_stars(addr, ra, dec, radius, tagalong)
static PyObject* starkd_search_stars(PyObject* self, PyObject* args) {
    startree_t* s;
    double ra, dec, radius;
    int N;
    PyObject* pyxyz;
    PyObject* pyradec;
    PyObject* pyinds;
    double* xyzres = NULL;
    double* radecres = NULL;
    int* inds = NULL;
    unsigned char tag;
    int i, C;
    PyObject* pydict;

    if (!PyArg_ParseTuple(args, "ldddb", &s, &ra, &dec, &radius, &tag)) {
        PyErr_SetString(PyExc_ValueError, "need four args: starkd, ra, dec, radius");
        return NULL;
    }

    //printf("RA,Dec,radius (%g,%g), %g\n", ra, dec, radius);
    startree_search_for_radec(s, ra, dec, radius, &xyzres, &radecres, &inds, &N);
    //printf("Found %i; xyz %p, radecs %p, inds %p\n", N, xyzres, radecres, inds);

    assert(N == 0 || xyzres);
    assert(N == 0 || radecres);
    assert(N == 0 || inds);

    //printf("pyxyz...\n");
    pyxyz = arrayd_to_pylist2(xyzres, N, 3);
    //printf("done.\n");
    //printf("pyradec...\n");
    pyradec = arrayd_to_pylist2(radecres, N, 2);
    //printf("done.\n");
    //printf("pyinds...\n");
    pyinds = arrayi_to_pylist(inds, N);
    //printf("done.\n");

    if (!tag)
        return Py_BuildValue("(OOO)", pyxyz, pyradec, pyinds);

    if (!startree_has_tagalong(s) || N == 0) {
        return Py_BuildValue("(OOOO)", pyxyz, pyradec, pyinds, PyDict_New());
    }
    C = startree_get_tagalong_N_columns(s);
    pydict = PyDict_New();
    for (i=0; i<C; i++) {
        PyObject* pyval;
        void* vdata;
        const char* name;
        tfits_type ft, readtype;
        int arr;
        PyObject* (*pyvalfunc)(void* v, int i);

        name = startree_get_tagalong_column_name(s, i);
        ft = startree_get_tagalong_column_fits_type(s, i);
        readtype = ft;

        switch (ft) {
            /*
             // FIXME -- special-case this?
             case TFITS_BIN_TYPE_A:
             case TFITS_BIN_TYPE_B:
             case TFITS_BIN_TYPE_X:
             // these are actually the characters 'T' and 'F'.
             case TFITS_BIN_TYPE_L:
             pytype = NPY_BYTE;
             break;
             */
        case TFITS_BIN_TYPE_D:
        case TFITS_BIN_TYPE_E:
            readtype = TFITS_BIN_TYPE_D;
            pyvalfunc = pyval_double;
            break;
        case TFITS_BIN_TYPE_I:
        case TFITS_BIN_TYPE_J:
            readtype = TFITS_BIN_TYPE_J;
            pyvalfunc = pyval_int;
            break;
        case TFITS_BIN_TYPE_K:
            readtype = TFITS_BIN_TYPE_K;
            pyvalfunc = pyval_int64;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "failed to map FITS type %i to numpy type, for column \"%s\"", ft, name);
            return NULL;
        }

        //int arr = startree_get_tagalong_column_array_size(s, i);
        vdata = fitstable_read_column_array_inds(startree_get_tagalong(s), name, readtype, inds, N, &arr);
        if (!vdata) {
            PyErr_Format(PyExc_ValueError, "fail to read tag-along column \"%s\"", name);
            return NULL;
        }

        if (arr > 1)
            pyval = array_to_pylist2(vdata, N, arr, pyvalfunc);
        else
            pyval = array_to_pylist(vdata, N, pyvalfunc);

        if (PyDict_SetItemString(pydict, name, pyval)) {
            PyErr_Format(PyExc_ValueError, "fail to set tag-along column value, for \"%s\"", name);
            return NULL;
        }
    }

    return Py_BuildValue("(OOOO)", pyxyz, pyradec, pyinds, pydict);
}




// starkd_search_stars_numpy(addr, ra, dec, radius, tagalong)
static PyObject* starkd_search_stars_numpy(PyObject* self, PyObject* args) {
    startree_t* s;
    double ra, dec, radius;
    int N;
    PyArrayObject* pyxyz;
    PyArrayObject* pyradec;
    PyArrayObject* pyinds;
    npy_intp dims[2];
    double* xyzres;
    double* radecres;
    int* inds;
    unsigned char tag;
    int i, C;
    PyObject* pydict;

    if (!PyArg_ParseTuple(args, "ldddb", &s, &ra, &dec, &radius, &tag)) {
        PyErr_SetString(PyExc_ValueError, "need four args: starkd, ra, dec, radius");
        return NULL;
    }

    startree_search_for_radec(s, ra, dec, radius, &xyzres, &radecres, &inds, &N);

    dims[0] = N;
    dims[1] = 3;
    pyxyz = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, xyzres);

    dims[0] = N;
    dims[1] = 2;
    pyradec = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, radecres);

    dims[0] = N;
    dims[1] = 1;
    pyinds = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, PyArray_INT, inds);

    if (!tag)
        return Py_BuildValue("(OOO)", pyxyz, pyradec, pyinds);

    if (!startree_has_tagalong(s) || N == 0) {
        return Py_BuildValue("(OOOO)", pyxyz, pyradec, pyinds, PyDict_New());
    }
    C = startree_get_tagalong_N_columns(s);
    pydict = PyDict_New();
    for (i=0; i<C; i++) {
        PyObject* pyval;
        void* vdata;
        const char* name;
        tfits_type ft, readtype;
        int nd;
        int arr;
        int pytype = 0;
        name = startree_get_tagalong_column_name(s, i);
        ft = startree_get_tagalong_column_fits_type(s, i);
        readtype = ft;
        switch (ft) {
            // FIXME -- special-case this?
        case TFITS_BIN_TYPE_A:
        case TFITS_BIN_TYPE_B:
        case TFITS_BIN_TYPE_X:
            // these are actually the characters 'T' and 'F'.
        case TFITS_BIN_TYPE_L:
            pytype = NPY_BYTE;
            break;
        case TFITS_BIN_TYPE_D:
            pytype = NPY_FLOAT64;
            break;
        case TFITS_BIN_TYPE_E:
            pytype = NPY_FLOAT32;
            break;
        case TFITS_BIN_TYPE_I:
            pytype = NPY_INT16;
            break;
        case TFITS_BIN_TYPE_J:
            pytype = NPY_INT32;
            break;
        case TFITS_BIN_TYPE_K:
            pytype = NPY_INT64;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "failed to map FITS type %i to numpy type, for column \"%s\"", ft, name);
            return NULL;
        }

        //int arr = startree_get_tagalong_column_array_size(s, i);
        vdata = fitstable_read_column_array_inds(startree_get_tagalong(s), name, readtype, inds, N, &arr);
        if (!vdata) {
            PyErr_Format(PyExc_ValueError, "fail to read tag-along column \"%s\"", name);
            return NULL;
        }
        dims[0] = N;
        dims[1] = arr;
        if (arr > 1)
            nd = 2;
        else
            nd = 1;
        pyval = PyArray_SimpleNewFromData(nd, dims, pytype, vdata);
        if (PyDict_SetItemString(pydict, name, pyval)) {
            PyErr_Format(PyExc_ValueError, "fail to set tag-along column value, for \"%s\"", name);
            return NULL;
        }
    }

    return Py_BuildValue("(OOOO)", pyxyz, pyradec, pyinds, pydict);
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
    { "starkd_search_stars", starkd_search_stars, METH_VARARGS,
      "Search for stars in a region; return as python lists." },
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
