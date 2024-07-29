/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include "Python.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#define PyInt_FromLong PyLong_FromLong
#endif

#include <stdio.h>
#include <assert.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#undef I

#include "os-features.h"
#include "keywords.h"
#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "dualtree_rangesearch.h"
#include "dualtree_nearestneighbour.h"
#include "bl.h"
#include "mathutil.h"
#include "errors.h"

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    int opened;
    kdtree_t* kd;
} KdObject;

static void KdTree_dealloc(KdObject* self) {
    //printf("dealloc for KdObject %p, opened %i, kd %p\n",
    //self, self->opened, self->kd);
    if (self && self->kd) {
        if (self->opened) {
            kdtree_fits_close(self->kd);
        } else {
            free(self->kd->data.any);
            kdtree_free(self->kd);
        }
        self->kd = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* KdTree_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    KdObject *self;

    self = (KdObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->opened = 0;
        self->kd = NULL;
    }
    return (PyObject*)self;
}

static int KdTree_init(KdObject *self, PyObject *args, PyObject *keywords) {
    Py_ssize_t n;
    PyArrayObject *x = NULL;
    char* filename = NULL;
    char* treename = NULL;
    VarUnused PyObject *fnbytes = NULL;

    static char *kwlist[] = {"data", "nleaf", "bbox", "split", NULL};

    int Nleaf = 16;
    int do_bbox = 1;
    int do_split = 0;
    
    n = PyTuple_Size(args);
    if (!((n == 1) || (n == 2))) {
        PyErr_SetString(PyExc_ValueError, "need one or two args: (array x), or (kdtree filename, + optionally tree name)");
        return -1;
    }
    
    // Try parsing as an array.
    if ((n == 1) &&
        PyArg_ParseTupleAndKeywords(args, keywords, "O!|ipp", kwlist,
                                    &PyArray_Type, &x, &Nleaf, &do_bbox,
                                    &do_split)) {
        // kdtree_build
        int treeoptions, treetype;
        int N, D;
        int i,j;
        double* data;
        uint64_t* udata;
        void* kd_data;
        anbool isdouble = TRUE;

        self->opened = 0;
        if (PyArray_NDIM(x) != 2) {
            PyErr_SetString(PyExc_ValueError, "array must be two-dimensional");
            return -1;
        }


        if (PyArray_TYPE(x) == NPY_DOUBLE) {
            treetype = KDTT_DOUBLE;
            isdouble = TRUE;
        } else if (PyArray_TYPE(x) == NPY_UINT64) {
            treetype = KDTT_U64;
            isdouble = FALSE;
        } else {
            PyErr_SetString(PyExc_ValueError, "array must contain doubles or uint64s");
            return -1;
        }

        N = (int)PyArray_DIM(x, 0);
        D = (int)PyArray_DIM(x, 1);
        if (D > 10) {
            PyErr_SetString(PyExc_ValueError, "maximum dimensionality is 10: maybe you need to transpose your array?");
            return -1;
        }
        if (!do_bbox && !do_split) {
            PyErr_SetString(PyExc_ValueError, "need to set bbox=True or split=True");
            return -1;
        }
        // FIXME -- should be able to do this faster...
        if (isdouble) {
            data = malloc(N * D * sizeof(double));
            kd_data = data;
            for (i=0; i<N; i++) {
                for (j=0; j<D; j++) {
                    double* pd = PyArray_GETPTR2(x, i, j);
                    data[i*D + j] = *pd;
                }
            }
        } else {
            udata = malloc(N * D * sizeof(uint64_t));
            kd_data = udata;
            for (i=0; i<N; i++) {
                for (j=0; j<D; j++) {
                    uint64_t* pd = PyArray_GETPTR2(x, i, j);
                    udata[i*D + j] = *pd;
                }
            }
        }
        treeoptions = 0;
        if (do_bbox)
            treeoptions |= KD_BUILD_BBOX;
        if (do_split)
            treeoptions |= KD_BUILD_SPLIT;
        self->kd = kdtree_build(NULL, kd_data, N, D, Nleaf, treetype, treeoptions);
        if (!self->kd)
            return -1;
        return 0;
    }

    // clear the exception from PyArg_ParseTuple above
    PyErr_Clear();

    // kdtree_fits_open
    self->opened = 1;

#if defined(IS_PY3K)
    if (n == 1) {
        if (!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &fnbytes))
            return -1;
    } else {
        if (!PyArg_ParseTuple(args, "O&s", PyUnicode_FSConverter, &fnbytes, &treename))
            return -1;
    }
    if (fnbytes == NULL)
        return -1;
    filename = PyBytes_AsString(fnbytes);
#else
    if (n == 1) {
        if (!PyArg_ParseTuple(args, "s", &filename))
            return -1;
    } else {
        if (!PyArg_ParseTuple(args, "ss", &filename, &treename))
            return -1;
    }
#endif

    char* errstr = NULL;
    errors_start_logging_to_string();
    self->kd = kdtree_fits_read(filename, treename, NULL);
    errstr = errors_stop_logging_to_string("\n");

    if (!self->kd) {
        if (fnbytes && ((errno == ENOENT) || (errno == EACCES) || (errno == EEXIST))) {
            PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, fnbytes);
            Py_DECREF(fnbytes);
            return -1;
        }
        PyErr_SetString(PyExc_ValueError, errstr);
        if (fnbytes)
            Py_DECREF(fnbytes);
        return -1;
    }
    if (fnbytes)
        Py_DECREF(fnbytes);
    return 0;
}

static PyObject* KdTree_set_name(KdObject* self, PyObject* args) {
    char* name = NULL;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: Kd-Tree name (string)");
        return NULL;
    }
    if (self->kd->name)
        free(self->kd->name);
    self->kd->name = strdup(name);
    Py_RETURN_NONE;
}

static PyObject* KdTree_write(KdObject* self, PyObject* args) {
    char* fn;
    int rtn;
    
#if defined(IS_PY3K)
    PyObject *fnbytes = NULL;
    if (!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &fnbytes)) {
        return NULL;
    }
    if (fnbytes == NULL)
        return NULL;
    fn = PyBytes_AsString(fnbytes);
    rtn = kdtree_fits_write(self->kd, fn, NULL);
    Py_DECREF(fnbytes);
#else
    if (!PyArg_ParseTuple(args, "s", &fn)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: filename (string)");
        return NULL;
    }
    rtn = kdtree_fits_write(self->kd, fn, NULL);
#endif
    return Py_BuildValue("i", rtn);
}

static PyObject* KdTree_print(KdObject* self) {
    kdtree_print(self->kd);
    Py_RETURN_NONE;
}

static PyObject* KdTree_search(KdObject* self, PyObject* args) {
    void* X;
    PyObject* rtn;
    npy_intp dims[1];
    kdtree_t* kd;
    int D, N;
    PyObject* pyO;
    PyArrayObject* npI;
    PyObject* pyInds;
    PyObject* pyDists = NULL;
    PyArray_Descr* dtype;
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED
        | NPY_ARRAY_ELEMENTSTRIDES;
    double radius;
    kdtree_qres_t* res;
    int getdists, sortdists;
    int opts;

    if (!PyArg_ParseTuple(args, "Odii",
                          &pyO, &radius,
                          &getdists, &sortdists)) {
        PyErr_SetString(PyExc_ValueError, "need four args: query point (numpy array of floats), radius (double), get distances (int 0/1), sort distances (int 0/1)");
        return NULL;
    }
    kd = self->kd;
    D = kd->ndim;

    if (sortdists) {
        getdists = 1;
    }

    if (kdtree_exttype(kd) == KDT_EXT_U64)
        dtype = PyArray_DescrFromType(NPY_UINT64);
    else
        dtype = PyArray_DescrFromType(NPY_DOUBLE);

    Py_INCREF(dtype);
    npI = (PyArrayObject*)PyArray_FromAny(pyO, dtype, 1, 1, req, NULL);
    if (!npI) {
        PyErr_SetString(PyExc_ValueError, "Failed to convert query point array to np array of float or uint64 (depending on tree data type)");
        Py_XDECREF(dtype);
        return NULL;
    }
    N = (int)PyArray_DIM(npI, 0);
    if (N != D) {
        PyErr_SetString(PyExc_ValueError, "Query point must have size == dimension of tree");
        Py_DECREF(npI);
        Py_DECREF(dtype);
        return NULL;
    }

    X = PyArray_DATA(npI);

    opts = 0;
    if (getdists) {
        opts |= KD_OPTIONS_COMPUTE_DISTS;
    }
    if (sortdists) {
        opts |= KD_OPTIONS_SORT_DISTS;
    }

    res = kdtree_rangesearch_options(kd, X, radius*radius, opts);
    N = res->nres;
    dims[0] = N;
    res->inds = realloc(res->inds, N * sizeof(uint32_t));
    pyInds = PyArray_SimpleNewFromData(1, dims, NPY_UINT32, res->inds);
    res->inds = NULL;

    if (getdists) {
        res->sdists = realloc(res->sdists, N * sizeof(double));
        pyDists = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, res->sdists);
        res->sdists = NULL;
    }

    kdtree_free_query(res);

    Py_DECREF(npI);
    Py_DECREF(dtype);
    if (getdists) {
        rtn = Py_BuildValue("(OO)", pyInds, pyDists);
        Py_DECREF(pyDists);
    } else {
        rtn = Py_BuildValue("O", pyInds);
    }
    Py_DECREF(pyInds);
    return rtn;
}

static PyObject* KdTree_get_data(KdObject* self, PyObject* args) {
    PyArrayObject* pyX;
    PyObject* rtn;
    npy_intp dims[2];
    kdtree_t* kd;
    int k, D, N;
    npy_uint32* I;
    PyObject* pyO;
    PyArrayObject* npI;
    // this is the type returned by kdtree_rangesearch
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_UINT32);
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED
        | NPY_ARRAY_ELEMENTSTRIDES;

    if (!PyArg_ParseTuple(args, "O", &pyO)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: index array (numpy array of ints)");
        return NULL;
    }
    kd = self->kd;
    D = kd->ndim;

    Py_INCREF(dtype);
    npI = (PyArrayObject*)PyArray_FromAny(pyO, dtype, 1, 1, req, NULL);
    if (!npI) {
        PyErr_SetString(PyExc_ValueError, "Failed to convert index array to np array of uint32");
        Py_XDECREF(dtype);
        return NULL;
    }
    N = (int)PyArray_DIM(npI, 0);

    dims[0] = N;
    dims[1] = D;

    I = PyArray_DATA(npI);
    if (kdtree_datatype(kd) == KDT_DATA_U64) {
        uint64_t* X;
        pyX = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT64);
        X = PyArray_DATA(pyX);
        for (k=0; k<N; k++) {
            memcpy(X, kdtree_get_data(kd, I[k]), D*sizeof(uint64_t));
            X += D;
        }
        Py_DECREF(npI);
    } else {
        double* X;
        pyX = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        X = PyArray_DATA(pyX);
        for (k=0; k<N; k++) {
            kdtree_copy_data_double(kd, I[k], 1, X);
            X += D;
        }
        Py_DECREF(npI);
    }
    Py_DECREF(dtype);
    rtn = Py_BuildValue("O", pyX);
    Py_DECREF(pyX);
    return rtn;
}

static PyObject* KdTree_permute(KdObject* self, PyObject* args) {
    PyArrayObject* pyX;
    npy_int* X;
    PyObject* rtn;
    npy_intp dims[1];
    kdtree_t* kd;
    long k, N;
    npy_int* I;
    PyObject* pyO;
    PyArrayObject* npI;
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_INT);
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED
        | NPY_ARRAY_ELEMENTSTRIDES;

    if (!PyArg_ParseTuple(args, "O", &pyO)) {
        PyErr_SetString(PyExc_ValueError, "need one arg: index array (numpy array of ints)");
        return NULL;
    }
    kd = self->kd;

    Py_INCREF(dtype);
    npI = (PyArrayObject*)PyArray_FromAny(pyO, dtype, 1, 1, req, NULL);
    if (!npI) {
        PyErr_SetString(PyExc_ValueError, "Failed to convert index array to np array of int");
        Py_XDECREF(dtype);
        return NULL;
    }
    N = PyArray_DIM(npI, 0);

    dims[0] = N;

    pyX = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    X = PyArray_DATA(pyX);
    I = PyArray_DATA(npI);

    for (k=0; k<N; k++) {
        npy_int ii = I[k];
        //printf("Permute: ii=%i\n", ii);
        X[k] = kdtree_permute(kd, ii);
    }
    Py_DECREF(npI);
    Py_DECREF(dtype);
    rtn = Py_BuildValue("O", pyX);
    Py_DECREF(pyX);
    return rtn;
}

static PyMethodDef kdtree_methods[] = {
    {"set_name", (PyCFunction)KdTree_set_name, METH_VARARGS,
     "Sets the Kd-Tree's name to the given string",
    },
    {"write", (PyCFunction)KdTree_write, METH_VARARGS,
     "Writes the Kd-Tree to the given (string) filename in FITS format."
    },
    {"print", (PyCFunction)KdTree_print, METH_NOARGS,
     "Prints a representation of the Kd-Tree to stdout."
    },
    {"search", (PyCFunction)KdTree_search, METH_VARARGS,
     "Searches for points within range in the Kd-Tree."
    },
    {"get_data", (PyCFunction)KdTree_get_data, METH_VARARGS,
     "Returns data from this tree, given numpy array of indices (MUST be np.uint32)."
    },
    {"permute", (PyCFunction)KdTree_permute, METH_VARARGS,
     "Applies this Kd-Tree's permutation to the given numpy array of integers (to get from Kd-Tree indices back to the original indexing."
    },
    {NULL}
};

static PyObject* KdTree_n(KdObject* self, void* closure) {
    return PyInt_FromLong(kdtree_n(self->kd));
}

static PyObject* KdTree_bbox(KdObject* self, void* closure) {
    PyArrayObject* bbox;
    PyObject* rtn;
    npy_intp dims[2];
    double *bb;
    anbool ok;
    kdtree_t* kd;
    int j, D;
    kd = self->kd;
    D = kd->ndim;
    dims[0] = D;
    dims[1] = 2;
    bbox = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    {
        double bblo[D];
        double bbhi[D];
        ok = kdtree_get_bboxes(kd, 0, bblo, bbhi);
        if (!ok) {
            Py_RETURN_NONE;
        }
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

static PyGetSetDef kdtree_getseters[] = {
    {"n",
     (getter)KdTree_n, NULL, "number of data items in kd-tree",
     NULL},
    {"bbox",
     (getter)KdTree_bbox, NULL,
     "Returns a numpy array containing this Kd-Tree's bounding box in data space"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject KdType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "spherematch.KdTree",      /* tp_name */
    sizeof(KdObject),          /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)KdTree_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "KdTree object",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    kdtree_methods,            /* tp_methods */
    0, //Noddy_members,             /* tp_members */
    kdtree_getseters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)KdTree_init,     /* tp_init */
    0,                         /* tp_alloc */
    KdTree_new,                /* tp_new */
};

struct dualtree_results2 {
    kdtree_t *kd1;
    kdtree_t *kd2;
    PyObject* indlist;
    anbool permute;
};

static void callback_dualtree2(void* v, int ind1, int ind2, double dist2) {
    struct dualtree_results2* dt = v;
    PyObject* lst;
    if (dt->permute) {
        ind1 = kdtree_permute(dt->kd1, ind1);
        ind2 = kdtree_permute(dt->kd2, ind2);
    }
    lst = PyList_GET_ITEM(dt->indlist, ind1);
    if (!lst) {
        lst = PyList_New(1);
        // SetItem steals a ref -- that's what we want.
        PyList_SetItem(dt->indlist, ind1, lst);
        PyList_SET_ITEM(lst, 0, PyInt_FromLong(ind2));
    } else {
        PyList_Append(lst, PyInt_FromLong(ind2));
    }
}

static PyObject* spherematch_match2(PyObject* self, PyObject* args) {
    int i, N;
    struct dualtree_results2 dtresults;
    KdObject *kdobj1 = NULL, *kdobj2 = NULL;
    kdtree_t *kd1, *kd2;
    double rad;
    PyObject* indlist;
    anbool notself;
    anbool permute;
	
    // So that ParseTuple("b") with a C "anbool" works
    assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "O!O!dbb",
                          &KdType, &kdobj1, &KdType, &kdobj2,
                          &rad, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two KdTree objects, search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }
    kd1 = kdobj1->kd;
    kd2 = kdobj2->kd;
    
    N = kdtree_n(kd1);
    indlist = PyList_New(N);
    assert(indlist);

    dtresults.kd1 = kd1;
    dtresults.kd2 = kd2;
    dtresults.indlist = indlist;
    dtresults.permute = permute;

    dualtree_rangesearch(kd1, kd2, 0.0, rad, notself, NULL,
                         callback_dualtree2, &dtresults,
                         NULL, NULL);

    // set empty slots to None, not NULL.
    for (i=0; i<N; i++) {
        if (PyList_GET_ITEM(indlist, i))
            continue;
        Py_INCREF(Py_None);
        PyList_SET_ITEM(indlist, i, Py_None);
    }

    return indlist;
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
    size_t i, N;
    KdObject *kdobj1 = NULL, *kdobj2 = NULL;
    kdtree_t *kd1, *kd2;
    double rad;
    struct dualtree_results dtresults;
    PyArrayObject* inds;
    npy_intp dims[2];
    PyArrayObject* dists;
    anbool notself;
    anbool permute;
    PyObject* rtn;
	
    // So that ParseTuple("b") with a C "anbool" works
    assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "O!O!dbb",
                          &KdType, &kdobj1, &KdType, &kdobj2,
                          &rad, &notself, &permute)) {
        PyErr_SetString(PyExc_ValueError, "spherematch_c.match: need five args: two KdTree objects, search radius (float), notself (boolean), permuted (boolean)");
        return NULL;
    }
    kd1 = kdobj1->kd;
    kd2 = kdobj2->kd;

    dtresults.inds1 = il_new(256);
    dtresults.inds2 = il_new(256);
    dtresults.dists = dl_new(256);
    dualtree_rangesearch(kd1, kd2, 0.0, rad, notself, NULL,
                         callback_dualtree, &dtresults,
                         NULL, NULL);

    N = il_size(dtresults.inds1);
    dims[0] = N;
    dims[1] = 2;

    inds =  (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT);
    dims[1] = 1;
    dists = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    for (i=0; i<N; i++) {
        int index;
        int* iptr;
        double* dptr;
        iptr = PyArray_GETPTR2(inds, i, 0);
        index = il_get(dtresults.inds1, i);
        if (permute)
            index = kdtree_permute(kd1, index);
        *iptr = index;
        iptr = PyArray_GETPTR2(inds, i, 1);
        index = il_get(dtresults.inds2, i);
        if (permute)
            index = kdtree_permute(kd2, index);
        *iptr = index;
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
    KdObject *kdobj1 = NULL, *kdobj2 = NULL;
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

    if (!PyArg_ParseTuple(args, "O!O!db",
                          &KdType, &kdobj1, &KdType, &kdobj2,
                          &rad, &notself)) {
        PyErr_SetString(PyExc_ValueError, "need three args: two KdTree objects, and search radius");
        return NULL;
    }
    kd1 = kdobj1->kd;
    kd2 = kdobj2->kd;

    NY = kdtree_n(kd2);

    dims[0] = NY;
    inds   = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT);
    dist2s = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    assert(PyArray_ITEMSIZE(inds) == sizeof(int));
    assert(PyArray_ITEMSIZE(dist2s) == sizeof(double));

    // YUCK!
    tempinds = (int*)malloc(NY * sizeof(int));
    double* tempdists = (double*)malloc(NY * sizeof(double));

    pinds   = tempinds; //PyArray_DATA(inds);
    //pdist2s = PyArray_DATA(dist2s);
    pdist2s = tempdists;

    dualtree_nearestneighbour(kd1, kd2, rad*rad, &pdist2s, &pinds, NULL, notself);

    // now we have to apply kd1's permutation array!
    for (i=0; i<NY; i++)
        if (pinds[i] != -1)
            pinds[i] = kdtree_permute(kd1, pinds[i]);


    pinds = PyArray_DATA(inds);
    pdist2s = PyArray_DATA(dist2s);

    for (i=0; i<NY; i++) {
        pinds[i] = -1;
        pdist2s[i] = LARGE_VAL;
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

static PyObject* spherematch_nn2(PyObject* self, PyObject* args) {
    int i, j, NY, N;
    KdObject *kdobj1 = NULL, *kdobj2 = NULL;
    kdtree_t *kd1, *kd2;
    npy_intp dims[1];
    PyObject* I;
    PyObject* J;
    PyObject* dist2s;
    PyObject* counts = NULL;
    int *pi;
    int *pj;
    int *pc = NULL;
    double *pd;
    double rad;
    anbool notself;
    anbool docount;
    int* tempinds;
    int* tempcount = NULL;
    int** ptempcount = NULL;
    double* tempd2;
    PyObject* rtn;

    // So that ParseTuple("b") with a C "anbool" works
    assert(sizeof(anbool) == sizeof(unsigned char));

    if (!PyArg_ParseTuple(args, "O!O!dbb",
                          &KdType, &kdobj1, &KdType, &kdobj2,
                          &rad, &notself, &docount)) {
        PyErr_SetString(PyExc_ValueError, "need five args: two kdtree identifiers (ints), search radius, notself (bool) and docount (bool)");
        return NULL;
    }
    kd1 = kdobj1->kd;
    kd2 = kdobj2->kd;

    // quick check for no-overlap case
    if (kdtree_node_node_mindist2_exceeds(kd1, 0, kd2, 0, rad*rad)) {
        // allocate empty return arrays
        dims[0] = 0;
        I = PyArray_SimpleNew(1, dims, NPY_INT);
        J = PyArray_SimpleNew(1, dims, NPY_INT);
        dist2s = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	if (docount) {
            counts = PyArray_SimpleNew(1, dims, NPY_INT);
            rtn = Py_BuildValue("(OOOO)", I, J, dist2s, counts);
            Py_DECREF(counts);
	} else {
            rtn = Py_BuildValue("(OOO)", I, J, dist2s);
	}
        Py_DECREF(I);
        Py_DECREF(J);
        Py_DECREF(dist2s);
        return rtn;
    }

    NY = kdtree_n(kd2);

    tempinds = (int*)malloc(NY * sizeof(int));
    tempd2 = (double*)malloc(NY * sizeof(double));
    if (docount) {
	tempcount = (int*)calloc(NY, sizeof(int));
        ptempcount = &tempcount;
    }

    dualtree_nearestneighbour(kd1, kd2, rad*rad, &tempd2, &tempinds, ptempcount, notself);

    // count number of matches
    N = 0;
    for (i=0; i<NY; i++)
        if (tempinds[i] != -1)
            N++;

    // allocate return arrays
    dims[0] = N;
    I = PyArray_SimpleNew(1, dims, NPY_INT);
    J = PyArray_SimpleNew(1, dims, NPY_INT);
    dist2s = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    pi = PyArray_DATA((PyArrayObject*)I);
    pj = PyArray_DATA((PyArrayObject*)J);
    pd = PyArray_DATA((PyArrayObject*)dist2s);
    if (docount) {
        counts = PyArray_SimpleNew(1, dims, NPY_INT);
        pc = PyArray_DATA((PyArrayObject*)counts);
    }

    j = 0;
    for (i=0; i<NY; i++) {
        if (tempinds[i] == -1)
            continue;
        pi[j] = kdtree_permute(kd1, tempinds[i]);
        pj[j] = kdtree_permute(kd2, i);
        pd[j] = tempd2[i];
        if (docount)
            pc[j] = tempcount[i];
        j++;
    }

    free(tempinds);
    free(tempd2);
    free(tempcount);

    if (docount) {
        rtn = Py_BuildValue("(OOOO)", I, J, dist2s, counts);
        Py_DECREF(counts);
    } else {
        rtn = Py_BuildValue("(OOO)", I, J, dist2s);
    }
    Py_DECREF(I);
    Py_DECREF(J);
    Py_DECREF(dist2s);
    return rtn;
}

static PyMethodDef spherematchMethods[] = {
    { "match", spherematch_match, METH_VARARGS,
      "find matching data points" },
    { "match2", spherematch_match2, METH_VARARGS,
      "find matching data points" },
    { "nearest", spherematch_nn, METH_VARARGS,
      "find nearest neighbours" },
    { "nearest2", spherematch_nn2, METH_VARARGS,
      "find nearest neighbours (different return values)" },
    {NULL, NULL, 0, NULL}
};

#if defined(IS_PY3K)

static struct PyModuleDef spherematch_module = {
    PyModuleDef_HEAD_INIT,
    "spherematch_c",
    NULL,
    0,
    spherematchMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_spherematch_c(void) {
    PyObject *m;
    import_array();

    KdType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&KdType) < 0)
        return NULL;

    m = PyModule_Create(&spherematch_module);
    if (m == NULL)
        return NULL;

    Py_INCREF((PyObject*)&KdType);
    PyModule_AddObject(m, "KdTree", (PyObject*)&KdType);

    return m;
}

#else

PyMODINIT_FUNC
initspherematch_c(void) {
    PyObject* m;
    import_array();

    KdType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&KdType) < 0)
        return;

    m = Py_InitModule3("spherematch_c", spherematchMethods,
                       "spherematch_c provides python bindings for the libkd library");

    Py_INCREF((PyObject*)&KdType);
    PyModule_AddObject(m, "KdTree", (PyObject*)&KdType);
}

#endif
