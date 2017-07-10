/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#define GLUE2(a,b) a ## b
#define GLUE(a,b) GLUE2(a, b)

int GLUE(dsigma, DSIGMA_SUFF)(IMGTYPE *image,
                              int nx,
                              int ny,
                              int sp,
                              int gridsize,
                              float *sigma) {

#undef GLUE
#undef GLUE2

    float *diff = NULL;
    float tot;
    int i, j, n, dx, dy, ndiff;
    int rtn = 0;

    if (nx == 1 && ny == 1) {
        *sigma = 0.;
        return rtn;
    }

    if (gridsize == 0)
        gridsize = 20;

    dx = gridsize;
    if (dx > nx / 4)
        dx = nx / 4;
    if (dx <= 0)
        dx = 1;

    dy = gridsize;
    if (dy > ny / 4)
        dy = ny / 4;
    if (dy <= 0)
        dy = 1;

    /* get a bunch of noise 'samples' by looking at the differences between two
     * diagonally spaced pixels (usually 5) */
    ndiff = ((nx-sp + dx-1)/dx) * ((ny-sp + dy-1)/dy);

    if (ndiff <= 1) {
        *sigma = 0.;
        return rtn;
    }

    logverb("Sampling sigma at %i points\n", ndiff);
    diff = malloc(ndiff * sizeof(float));
    n = 0;
    for (j = 0; j < ny-sp; j += dy) {
        for (i = 0; i < nx-sp; i += dx) {
            //diff[n] = fabs(image[i + j * nx] - image[i + sp + (j + sp) * nx]);
            diff[n] = fabs((float)image[i + j * nx] - (float)image[i + sp + (j + sp) * nx]);
            n++;
        }
    }
    assert(n == ndiff);

    if (ndiff <= 10) {
        tot = 0.;
        for (i = 0; i < ndiff; i++)
            tot += diff[i] * diff[i];
        *sigma = sqrt(tot / (float) ndiff);
        goto finish;
    }

    /*
     estimate sigma in a clever way to avoid having our estimate
     biased by outliers. outliers come into the diff list when we
     sampled a point where the upper point was on a source, but the
     lower one was not (or vice versa).  Since the sample variance
     involves squaring the already-large outliers, they drastically
     affect the final sigma estimate. by sorting, the outliers go to
     the top and only affect the final value very slightly, because
     they are a small fraction of the total entries in diff (or so we
     hope!)
     */

    {
        double Nsigma=0.7;
        double s = 0.0;
        // Sample the sorted list of squared differences at different
        // percentiles (starting at ~50th)
        while (s == 0.0) {
            int k = (int)floor(ndiff * erf(Nsigma / M_SQRT2));
            if (k >=  ndiff) {
                logerr("Failed to estimate the image noise.  Setting sigma=1.  Expect the worst.\n");
                // FIXME - Could try a finer grid of sample points...
                s = 1.0;
                break;
            }
            s = dselip(k, ndiff, diff) / (Nsigma * M_SQRT2);
            logverb("Nsigma=%g, s=%g\n", Nsigma, s);
            Nsigma += 0.1;
        }
        *sigma = s;
    }
    rtn = 1;

 finish:
    FREEVEC(diff);
    return rtn;
} /* end dsigma */
