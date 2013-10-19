static int LANCZOS_INTERP_FUNC(PyObject* np_ixi, PyObject* np_iyi,
                               PyObject* np_dx, PyObject* np_dy,
                               PyObject* loutputs, PyObject* linputs) {
    npy_intp W,H, N;
    npy_intp Nimages;
    npy_intp i, j;
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_FLOAT);
    PyArray_Descr* itype = PyArray_DescrFromType(NPY_INT32);
    int req = NPY_C_CONTIGUOUS | NPY_ALIGNED |
        NPY_NOTSWAPPED | NPY_ELEMENTSTRIDES;
    int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;

    const int32_t *ixi, *iyi;
    const float *dx, *dy;

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

    //static const int L = 5;
    // Nlutunit is number of bins per unit x
    //static const int Nlutunit = 1024;
    static const int Nlutunit = 2048;
    static const double lut0 = -(L + 0.5); //-5.5; //-(L+0.5);
    static const int Nunits = 2*(L+1); //12; //2*(L+1);
    //static const int Nlut = Nunits * Nlutunit;
    //static float lut[24576];
    //static float lut[Nunits*Nlutunit];
    // HACK -- 2048 here == Nlutunit... some versions of gcc don't believe Nunits*Nlutunit is constant
    static float lut[2*(L+1)*2048];
    static int initialized = 0;

    if (!initialized) {
        // this table has the elements you need to use together
        // stored together: L(x[0]), L(x[0]+1), L(x[0]+2), ...;
        // L(x[1]), L(x[1]+1), L(x[2]+2), ...
        for (i=0; i<Nlutunit; i++) {
            double x,f;
            double acc = 0.;
            x = (lut0 + ((i+0.5) / (double)Nlutunit));
            for (j=0; j<Nunits; j++, x+=1.0) {
                if (x <= -L || x >= L) {
                    f = 0.0;
                } else if (x == 0) {
                    f = 1.0;
                } else {
                    f = L * sin(M_PI * x) * sin(M_PI / L * x) / 
                        (M_PI * M_PI * x * x);
                }
                lut[i * Nunits + j] = f;
                acc += f;
            }
            lut[i*Nunits + Nunits-1] = acc;
            //printf("acc: %f\n", acc);
        }
        /*
         for (i=0; i<Nlut; i++) {
         printf("lut[% 4li] = %f\n", i, lut[i]);
         }
         */
        initialized = 1;
    }

    // CheckFromAny steals the dtype reference
    Py_INCREF(itype);
    np_ixi = PyArray_CheckFromAny(np_ixi, itype, 1, 1, req, NULL);
    np_iyi = PyArray_CheckFromAny(np_iyi, itype, 1, 1, req, NULL);
    // At this point, itype refcount = 0
    Py_INCREF(dtype);
    Py_INCREF(dtype);
    np_dx  = PyArray_CheckFromAny(np_dx,  dtype, 1, 1, req, NULL);
    np_dy  = PyArray_CheckFromAny(np_dy,  dtype, 1, 1, req, NULL);
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
        PyObject* np_inimg;
        PyObject* np_outimg;
        const float *inimg;
        float *outimg;

        ixi = PyArray_DATA(np_ixi);
        iyi = PyArray_DATA(np_iyi);
        dx  = PyArray_DATA(np_dx);
        dy  = PyArray_DATA(np_dy);

        np_inimg  = PyList_GetItem(linputs,  i);
        np_outimg = PyList_GetItem(loutputs, i);
        Py_INCREF(dtype);
        Py_INCREF(dtype);
        np_inimg  = PyArray_CheckFromAny(np_inimg,   dtype, 2, 2, req, NULL);
        np_outimg  = PyArray_CheckFromAny(np_outimg, dtype, 1, 1, reqout, NULL);
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
            tx0 = (int)((-(dx[j]+L) - lut0) * Nlutunit);
            ty0 = (int)((-(dy[j]+L) - lut0) * Nlutunit);
            // clip
            tx0 = MAX(0, MIN(Nlutunit-1, tx0));
            ty0 = MAX(0, MIN(Nlutunit-1, ty0));
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
                        accx  += (*lx) * (inpix[clipix]);
                    }
                    acc  += (*ly) * accx;
                }
            } else {
                iy -= L;
                // Lanczos kernel in y direction
                for (v=0; v<2*L+1; v++,
                         iy++, ly++) {
                    float accx = 0;
                    int ix = *ixi - L;
                    const float* lx = lut + tx0;
                    const float* inpix = inimg + iy * W + ix;
                    // Lanczos kernel in x direction
                    for (u=0; u<2*L+1; u++,
                             lx++, inpix++) {
                        accx  += (*lx) * (*inpix);
                    }
                    acc  += (*ly) * accx;
                }
            }
            nacc = lut[tx0 + Nunits-1] * lut[ty0 + Nunits-1];
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
