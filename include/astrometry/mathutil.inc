/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <math.h>

InlineDefine void normalize(double* x, double* y, double* z) {
	double invl = 1.0 / sqrt((*x)*(*x) + (*y)*(*y) + (*z)*(*z));
	*x *= invl;
	*y *= invl;
	*z *= invl;
}

InlineDefine void normalize_3(double* xyz) {
	double invlen = 1.0 / sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
	xyz[0] *= invlen;
	xyz[1] *= invlen;
	xyz[2] *= invlen;
}

InlineDefine void cross_product(double* a, double* b, double* cross) {
	cross[0] = a[1] * b[2] - a[2] * b[1];
	cross[1] = a[2] * b[0] - a[0] * b[2];
	cross[2] = a[0] * b[1] - a[1] * b[0];
}

InlineDefine int imax(int a, int b) {
	return (a > b) ? a : b;
}

InlineDefine int imin(int a, int b) {
	return (a < b) ? a : b;
}

InlineDefine double distsq_exceeds(double* d1, double* d2, int D, double limit) {
    double dist2;
    int i;
    dist2 = 0.0;
    for (i=0; i<D; i++) {
		dist2 += square(d1[i] - d2[i]);
		if (dist2 > limit)
			return 1;
    }
	return 0;
}

InlineDefine double distsq(const double* d1, const double* d2, int D) {
    double dist2;
    int i;
    dist2 = 0.0;
    for (i=0; i<D; i++) {
		dist2 += square(d1[i] - d2[i]);
    }
    return dist2;
}

InlineDefine double square(double d) {
	return d*d;
}

InlineDefine int inrange(double ra, double ralow, double rahigh) {
    if (ralow < rahigh) {
		if (ra >= ralow && ra <= rahigh)
            return 1;
        return 0;
    }

    /* handle wraparound properly */
    //if (ra <= ralow && ra >= rahigh)
    if (ra >= ralow || ra <= rahigh)
        return 1;
    return 0;
}
