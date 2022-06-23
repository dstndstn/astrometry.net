/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */

// L must be defined before including this file; L=3 or L=5 in practice.

// Preprocessor magic to glue together function names with L (3 or 5)
#define MOREGLUE(x, y) x ## y
#define GLUE(x, y) MOREGLUE(x, y)
// and for 3 tokens
#define MOREGLUE3(x, y, z) x ## y ## z
#define GLUE3(x, y, z) MOREGLUE3(x, y, z)

// lanczos_kernelf_[L]
static
float GLUE(lanczos_kernelf_, L)(float x) {
    static const float pif = M_PI;
    static const float pi2f = M_PI * M_PI;
    if (x <= -L || x >= L)
        return 0.0;
    if (x == 0)
        return 1.0;
    return L * sinf(pif * x) * sinf(pif / L * x) / (pi2f * x * x);
}
#define lanczos_kernelf(L, x) GLUE(lanczos_kernelf_, L)(x)

// Nlutunit is number of bins per unit x
// NOTE that this is share across different L values.
#ifndef LANCZOS_NLUT
#define LANCZOS_NLUT 1024
#endif

// We add an extra row to LANCZOS_NLUT so that we can compute the
// slope across each bin.
static float (GLUE(lut_, L))[2*(L+1)*(LANCZOS_NLUT+1)];
// Have we initialized the Look-up Table?
static int GLUE(lut_initialized_,L) = 0;

// Init look-up table
static void GLUE(lut_init_, L)() {
    if (GLUE(lut_initialized_,L))
        return;
    /*
     dx,dy are in [-0.5, 0.5].

     Lanczos-3 kernel is zero outside [-3,3].

     We build a look-up table where [0] is L(-3.5).

     And organized so that:
     lut[0] = L(-3.5)
     lut[1] = L(-2.5)
     lut[2] = L(-1.5)
     lut[3] = L(-0.5)
     lut[4] = L( 0.5)
     lut[5] = L( 1.5)
     lut[6] = L( 2.5)
     lut[7] stores sum(lut[0:7])

     lut[8]  = L(-3.499)
     lut[9]  = L(-2.499)
     lut[10] = L(-1.499)
     ...
     ...
     lut[8184] = L(-2.501)
     lut[8185] = L(-1.501)
     lut[8186] = L(-0.501)
     ...

     This is annoying because [-3.5,3] and [3,3.5] are zero so we
     have to sum 7 elements rather than 6.  But the alternatives
     seem worse.

     LANCZOS_NLUT aka Nlutunit is number of bins per unit x.
     Nunits is the number of units, ie the support of the kernel.
     */
    static const float lut0 = -(L + 0.5);
    static const int Nunits = 2*(L+1);
    // this table has the elements you need to use together
    // stored together: L(x[0]), L(x[0]+1), L(x[0]+2), ...;
    // L(x[1]), L(x[1]+1), L(x[2]+2), ...
    int i, j;
    float* lut = GLUE(lut_,L);
    //for (i=0; i<=Nlutunit; i++) {
    for (i=0; i<=LANCZOS_NLUT; i++) {
        float x,f;
        float acc = 0.;
        //x = lut0 + i / (float)(Nlutunit);
        x = lut0 + i / (float)(LANCZOS_NLUT);
        for (j=0; j<Nunits; j++, x+=1.0) {
            f = lanczos_kernelf(L, x);
            lut[i * Nunits + j] = f;
            acc += f;
        }
        // last column contains the sum
        lut[i*Nunits + Nunits-1] = acc;
    }
}
#define lut_init(L) GLUE(lut_init_, L)()

static float GLUE(lanczos_resample_one_, L)
     (int ix,
      float dx,
      int iy,
      float dy,
      const float* inimg,
      const int W,
      const int H) {

    const float* lut = GLUE(lut_, L);
    const float lut0 = -(L + 0.5);
    const int Nunits = 2*(L+1);

    float acc = 0.;
    float accx;
    float nacc;
    const float* ly;
    float fx,fy;
    float slope, slopey;
    npy_intp u,v;
    int tx0, ty0;

    // float bin
    fx = (-(dx+L) - lut0) * LANCZOS_NLUT;
    fy = (-(dy+L) - lut0) * LANCZOS_NLUT;
    tx0 = (int)fx;
    ty0 = (int)fy;
    // clip int bins
    tx0 = MAX(0, MIN(LANCZOS_NLUT-1, tx0));
    ty0 = MAX(0, MIN(LANCZOS_NLUT-1, ty0));
    // what fraction of the way through the bin are we?
    fx = fx - tx0;
    fy = fy - ty0;

    // find start of LUT row for this bin.
    tx0 *= Nunits;
    ty0 *= Nunits;

    ly = lut + ty0;
    // special-case pixels near the image edges.
    // (this is the same code except for the "clip" checks on X,Y coords)
    if (ix < L || ix >= (W-L) || iy < L || iy >= (H-L)) {
        iy -= L;
        // Lanczos kernel in y direction
        for (v=0; v<2*L+1; v++, iy++, ly++) {
            int clipiy = MAX(0, MIN((int)(H-1), iy));
            int x = ix - L;
            const float* lx = lut + tx0;
            const float* inpix = inimg + clipiy * W;
            // Lanczos kernel in x direction
            accx = 0.;
            for (u=0; u<2*L+1; u++, x++, lx++) {
                int clipix = MAX(0, MIN((int)(W-1), x));
                slope = lx[Nunits] - (*lx);
                accx  += ((*lx) + slope*fx) * (inpix[clipix]);

                //printf("weighting input (%i, %i) by kernel %f\n",
                //clipix, clipiy, (*lx) + slope*fx);
            }
            slope = ly[Nunits] - (*ly);
            acc  += ((*ly) + slope*fy) * accx;
        }
    } else {
        iy -= L;
        // Lanczos kernel in y direction
        for (v=0; v<2*L+1; v++, iy++, ly++) {
            const float* lx = lut + tx0;
            const float* inpix = inimg + iy * W + ix - L;
            accx = 0;
            // Lanczos kernel in x direction
            for (u=0; u<2*L+1; u++, lx++, inpix++) {
                slope = lx[Nunits] - (*lx);
                accx  += ((*lx) + slope*fx) * (*inpix);

                //printf("weighting input (%i, %i) by kernel %f\n",
                //ix, clipiy, (*lx) + slope*fx);
            }
            slope = ly[Nunits] - (*ly);
            acc  += ((*ly) + slope*fy) * accx;
        }
    }
    // Compute the slope across the X,Y normalizers as well.
    slope = lut[tx0 + Nunits-1 + Nunits] - lut[tx0 + Nunits-1];
    slopey = lut[ty0 + Nunits-1 + Nunits] - lut[ty0 + Nunits-1];
    nacc = ((lut[tx0 + Nunits-1] + slope  * fx) *
            (lut[ty0 + Nunits-1] + slopey * fy));
    return acc / nacc;
}
#define lanczos_resample_one GLUE(lanczos_resample_one_, L)


static PyObject* GLUE3(lanczos, L, _interpolate)
     (PyObject* py_ixi, PyObject* py_iyi,
      PyObject* py_dx, PyObject* py_dy,
      PyObject* loutputs, PyObject* linputs) {
    npy_intp W,H, N;
    npy_intp Nimages;
    npy_intp i, j;
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
    PyArray_Descr* itype = PyArray_DescrFromType(NPY_INT32);
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
        NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
    int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_WRITEBACKIFCOPY;

    const int32_t *ixi, *iyi;
    const float *dx, *dy;

    PyArrayObject *np_ixi, *np_iyi, *np_dx, *np_dy;
    lut_init(L);

    // CheckFromAny steals the dtype reference
    Py_INCREF(itype);
    np_ixi = (PyArrayObject*)PyArray_CheckFromAny(py_ixi, itype, 1, 1, req, NULL);
    np_iyi = (PyArrayObject*)PyArray_CheckFromAny(py_iyi, itype, 1, 1, req, NULL);
    // At this point, itype refcount = 0
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    np_dx  = (PyArrayObject*)PyArray_CheckFromAny(py_dx,  dtype, 1, 1, req, NULL);
    np_dy  = (PyArrayObject*)PyArray_CheckFromAny(py_dy,  dtype, 1, 1, req, NULL);
    // dtype refcount = 1 (we use it more below)
    if (!np_ixi || !np_iyi) {
        PyErr_SetString(PyExc_ValueError, "ixi,iyi arrays are wrong type / shape");
        return NULL;
    }
    if (!np_dx || !np_dy) {
        PyErr_SetString(PyExc_ValueError, "dx,dy arrays are wrong type / shape");
        return NULL;
    }
    N = PyArray_DIM(np_ixi, 0);
    if ((PyArray_DIM(np_iyi, 0) != N) ||
        (PyArray_DIM(np_dx,  0) != N) ||
        (PyArray_DIM(np_dy,  0) != N)) {
        PyErr_SetString(PyExc_ValueError, "ixi,iyi,dx,dy arrays must be same size");
        return NULL;
    }

    if (!PyList_Check(loutputs) ||
        !PyList_Check(linputs)) {
        PyErr_SetString(PyExc_ValueError, "loutputs and linputs must be lists of np arrays");
        return NULL;
    }
    Nimages = PyList_Size(loutputs);
    if (PyList_Size(linputs) != Nimages) {
        PyErr_SetString(PyExc_ValueError, "loutputs and linputs must be same length");
        return NULL;
    }

    for (i=0; i<Nimages; i++) {
        PyArrayObject* np_inimg;
        PyArrayObject* np_outimg;
        const float *inimg;
        float *outimg;

        ixi = PyArray_DATA(np_ixi);
        iyi = PyArray_DATA(np_iyi);
        dx  = PyArray_DATA(np_dx);
        dy  = PyArray_DATA(np_dy);

        Py_INCREF(dtype);
        Py_INCREF(dtype);
        np_inimg  = (PyArrayObject*)PyArray_CheckFromAny(PyList_GetItem(linputs,  i), dtype, 2, 2, req, NULL);
        np_outimg = (PyArrayObject*)PyArray_CheckFromAny(PyList_GetItem(loutputs, i), dtype, 1, 1, reqout, NULL);
        if (!np_inimg || !np_outimg) {
            PyErr_SetString(PyExc_ValueError, "Failed to convert input and output images to right type/shape");
            return NULL;
        }
        if (PyArray_DIM(np_outimg, 0) != N) {
            PyErr_SetString(PyExc_ValueError, "Output image must be same shape as ixo");
            return NULL;
        }
        H = PyArray_DIM(np_inimg, 0);
        W = PyArray_DIM(np_inimg, 1);
        inimg  = PyArray_DATA(np_inimg);
        outimg = PyArray_DATA(np_outimg);

        for (j=0; j<N; j++, outimg++, ixi++, iyi++) {
            // resample inimg[ iyi[j] + dy[j], ixi[j] + dx[j] ]
            // to outimg[ j ]
            *outimg = lanczos_resample_one(*ixi, dx[j], *iyi, dy[j], inimg, W, H);
        }
        Py_DECREF(np_inimg);
        if (PyArray_ResolveWritebackIfCopy(np_outimg) == -1) {
            PyErr_SetString(PyExc_ValueError, "Failed to write-back output image array!");
            Py_DECREF(np_outimg);
            return NULL;
        }
        Py_DECREF(np_outimg);
    }
    Py_DECREF(dtype);
    Py_DECREF(np_ixi);
    Py_DECREF(np_iyi);
    Py_DECREF(np_dx);
    Py_DECREF(np_dy);
    Py_RETURN_NONE;
}

static PyObject* GLUE3(lanczos, L, _interpolate_grid)
     (float x0, float xstep,
      float y0, float ystep,
      PyObject* output_img,
      PyObject* input_img) {
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
        NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
    int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_WRITEBACKIFCOPY;

    lut_init(L);

    PyArrayObject* np_inimg;
    PyArrayObject* np_outimg;
    // CheckFromAny steals the dtype reference
    Py_INCREF(dtype);
    np_inimg  = (PyArrayObject*)PyArray_CheckFromAny(input_img, dtype, 2, 2, req, NULL);
    if (!np_inimg) {
        PyErr_SetString(PyExc_ValueError, "input_image must be a 2-d float32 array");
        Py_XDECREF(dtype);
        return NULL;
    }
    Py_INCREF(dtype);
    np_outimg = (PyArrayObject*)PyArray_CheckFromAny(output_img, dtype, 2, 2, reqout, NULL);
    if (!np_outimg) {
        PyErr_SetString(PyExc_ValueError, "output_image must be a 2-d float32 array");
        Py_XDECREF(np_inimg);
        Py_XDECREF(dtype);
        return NULL;
    }

    const float *inimg = PyArray_DATA(np_inimg);
    float *outimg = PyArray_DATA(np_outimg);
    const npy_intp H = PyArray_DIM(np_inimg, 0);
    const npy_intp W = PyArray_DIM(np_inimg, 1);
    const npy_intp outH = PyArray_DIM(np_outimg, 0);
    const npy_intp outW = PyArray_DIM(np_outimg, 1);
    npy_intp i, j;
    for (j=0; j<outH; j++) {
        float y = y0 + j*ystep;
        int iy = lrintf(y);
        float dy = y - iy;
        assert((dy >= -0.5) && (dy <= 0.5));

        // Beyond the top of the input image + margin
        if ((iy < -L) || (iy >= H+L))
            continue;

        for (i=0; i<outW; i++) {
            float x = x0 + i*xstep;
            int ix = lrintf(x);
            float dx = x - ix;
            assert((dx >= -0.5) && (dx <= 0.5));

            if ((ix < -L) || (ix >= W+L))
                continue;

            //printf("resampling input at (%.3f, %.3f)\n", x, y);
            outimg[j * outW + i] = lanczos_resample_one(ix, dx, iy, dy, inimg, W, H);
        }
    }
    Py_DECREF(np_inimg);
    Py_DECREF(dtype);
    if (PyArray_ResolveWritebackIfCopy(np_outimg) == -1) {
        PyErr_SetString(PyExc_ValueError, "Failed to write-back output image array!");
        Py_DECREF(np_outimg);
        return NULL;
    }
    Py_DECREF(np_outimg);
    Py_RETURN_NONE;
}

#undef MOREGLUE
#undef GLUE
#undef lanczos_kernelf
#undef LANCZOS_NLUT
#undef lut_init
#undef lanczos_resample_one
