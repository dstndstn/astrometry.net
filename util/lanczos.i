/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */
static int LANCZOS_INTERP_FUNC(PyObject* py_ixi, PyObject* py_iyi,
                               PyObject* py_dx, PyObject* py_dy,
                               PyObject* loutputs, PyObject* linputs) {
    npy_intp W,H, N;
    npy_intp Nimages;
    npy_intp i, j;
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
    PyArray_Descr* itype = PyArray_DescrFromType(NPY_INT32);
    int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED |
        NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
    int reqout = req | NPY_ARRAY_WRITEABLE | NPY_ARRAY_UPDATEIFCOPY;

    const int32_t *ixi, *iyi;
    const float *dx, *dy;

    PyArrayObject *np_ixi, *np_iyi, *np_dx, *np_dy;

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
     lut[7] is empty for padding
            actually, = sum(lut[0:7])

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
     */

    // L must be defined before including this file; L=3 or L=5 in practice.

    // Nlutunit is number of bins per unit x
    static const int Nlutunit = 1024;
    static float lut[2*(L+1)*(1024+1)];
    // HACK -- we repeat the constant here because some versions of gcc don't believe Nunits*Nlutunit is constant

    static const double lut0 = -(L + 0.5);
    static const int Nunits = 2*(L+1);
    static int initialized = 0;

    if (!initialized) {
        // this table has the elements you need to use together
        // stored together: L(x[0]), L(x[0]+1), L(x[0]+2), ...;
        // L(x[1]), L(x[1]+1), L(x[2]+2), ...
        for (i=0; i<=Nlutunit; i++) {
            double x,f;
            double acc = 0.;
            x = lut0 + i / (double)(Nlutunit);
            for (j=0; j<Nunits; j++, x+=1.0) {
                f = lanczos_kernel(L, x);
                lut[i * Nunits + j] = f;
                acc += f;
            }
            lut[i*Nunits + Nunits-1] = acc;
        }
        initialized = 1;
        /* Print JSON
         printf("{ \"lut\": [\n");
         for (i=0; i<=Nlutunit; i++) {
         printf("%s[", i?",\n":"");
         for (j=0; j<Nunits; j++)
         printf("%s%f", j?",":"", lut[i*Nunits+j]);
         printf("]");
         }
         printf("] }\n");
         */
    }

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
        ERR("ixi,iyi arrays are wrong type / shape\n");
        return -1;
    }
    if (!np_dx || !np_dy) {
        ERR("dx,dy arrays are wrong type / shape\n");
        return -1;
    }
    N = PyArray_DIM(np_ixi, 0);
    if ((PyArray_DIM(np_iyi, 0) != N) ||
        (PyArray_DIM(np_dx,  0) != N) ||
        (PyArray_DIM(np_dy,  0) != N)) {
        ERR("ixi,iyi,dx,dy arrays must be same size\n");
        return -1;
    }

    if (!PyList_Check(loutputs) ||
        !PyList_Check(linputs)) {
        ERR("loutputs and linputs must be lists of np arrays\n");
        return -1;
    }
    Nimages = PyList_Size(loutputs);
    if (PyList_Size(linputs) != Nimages) {
        ERR("loutputs and linputs must be same length\n");
        return -1;
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
            ERR("Failed to convert input and output images to right type/shape\n");
            return -1;
        }
        if (PyArray_DIM(np_outimg, 0) != N) {
            ERR("Output image must be same shape as ixo\n");
            return -1;
        }
        H = PyArray_DIM(np_inimg, 0);
        W = PyArray_DIM(np_inimg, 1);
        inimg  = PyArray_DATA(np_inimg);
        outimg = PyArray_DATA(np_outimg);

        for (j=0; j<N; j++, outimg++, ixi++, iyi++) {
            // resample inimg[ iyi[j] + dy[j], ixi[j] + dx[j] ]
            // to outimg[ j ]
            npy_intp u,v;
            int tx0, ty0;
            float acc = 0.;
            float nacc;
            const float* ly;
            int ix,iy;
            float fx,fy, rx, ry;
            float slope, slopey;
            fx = (-(dx[j]+L) - lut0) * (Nlutunit);
            fy = (-(dy[j]+L) - lut0) * (Nlutunit);
            tx0 = (int)fx;
            ty0 = (int)fy;
            // clip
            tx0 = MAX(0, MIN(Nlutunit-1, tx0));
            ty0 = MAX(0, MIN(Nlutunit-1, ty0));
            // what fraction of the way through the bin are we?
            rx = fx - tx0;
            ry = fy - ty0;

            tx0 *= Nunits;
            ty0 *= Nunits;
            ly = lut + ty0;
            iy = *iyi;
            ix = *ixi;
            // special-case pixels near the image edges.
            if (ix < L || ix >= (W-L) || iy < L || iy >= (H-L)) {
                iy -= L;
                // Lanczos kernel in y direction
                for (v=0; v<2*L+1; v++, iy++, ly++) {
                    float accx = 0;
                    int clipiy = MAX(0, MIN((int)(H-1), iy));
                    int ix = *ixi - L;
                    const float* lx = lut + tx0;
                    const float* inpix = inimg + clipiy * W;
                    // Lanczos kernel in x direction
                    for (u=0; u<2*L+1; u++, ix++, lx++) {
                        int clipix = MAX(0, MIN((int)(W-1), ix));
                        slope = lx[Nunits] - (*lx);
                        accx  += ((*lx) + slope*rx) * (inpix[clipix]);
                    }
                    slope = ly[Nunits] - (*ly);
                    acc  += ((*ly) + slope*ry) * accx;
                }
            } else {
                iy -= L;
                // Lanczos kernel in y direction
                for (v=0; v<2*L+1; v++, iy++, ly++) {
                    float accx = 0;
                    int ix = *ixi - L;
                    const float* lx = lut + tx0;
                    const float* inpix = inimg + iy * W + ix;
                    // Lanczos kernel in x direction
                    for (u=0; u<2*L+1; u++,
                             lx++, inpix++) {
                        slope = lx[Nunits] - (*lx);
                        accx  += ((*lx) + slope*rx) * (*inpix);
                    }
                    slope = ly[Nunits] - (*ly);
                    acc  += ((*ly) + slope*ry) * accx;
                }
            }
            // Compute the slope across the X,Y normalizers as well.
            slope = lut[tx0 + Nunits-1 + Nunits] - lut[tx0 + Nunits-1];
            slopey = lut[ty0 + Nunits-1 + Nunits] - lut[ty0 + Nunits-1];
            nacc = ((lut[tx0 + Nunits-1] + slope  * rx) *
                    (lut[ty0 + Nunits-1] + slopey * ry));
            *outimg = acc / nacc;
        }
        Py_DECREF(np_inimg);
        Py_DECREF(np_outimg);
    }
    Py_DECREF(dtype);
    Py_DECREF(np_ixi);
    Py_DECREF(np_iyi);
    Py_DECREF(np_dx);
    Py_DECREF(np_dy);
    return 0;
}
