/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <fenv.h>

#include "resample.h"
#include "os-features.h"
#include "mathutil.h"
#include "errors.h"
#include "log.h"

double lanczos(double x, int order) {
    if (x == 0)
        return 1.0;
    if (x > order || x < -order)
        return 0.0;
    return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
    /*
     feclearexcept(FE_ALL_EXCEPT);
     double rtn = order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
     if (fetestexcept(FE_DIVBYZERO)) {
     printf("DIVBYZERO: %f\n", x);
     }
     if (fetestexcept(FE_INEXACT)) {
     printf("INEXACT: %f\n", x);
     }
     if (fetestexcept(FE_INVALID)) {
     printf("INVALID: %f\n", x);
     }
     if (fetestexcept(FE_OVERFLOW)) {
     printf("OVERFLOW: %f\n", x);
     }
     if (fetestexcept(FE_UNDERFLOW)) {
     printf("UNDERFLOW: %f\n", x);
     }
     return rtn;
     */
}

#define MANGLEGLUE2(n,f) n ## _ ## f
#define MANGLEGLUE(n,f) MANGLEGLUE2(n,f)
#define MANGLE(func) MANGLEGLUE(func, numbername)

#define numbername f
#define number float
#include "resample.inc"
#undef numbername
#undef number

#define numbername d
#define number double
#include "resample.inc"
#undef numbername
#undef number

#undef MANGLEGLUE2
#undef MANGLEGLUE
#undef MANGLE
