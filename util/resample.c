#include <stdlib.h>
#include <math.h>
#include <sys/param.h>

#include "resample.h"

#include "mathutil.h"
#include "errors.h"
#include "log.h"

static double lanczos(double x, int order) {
	if (x == 0)
		return 1.0;
	if (x > order || x < -order)
		return 0.0;
	return order * sin(M_PI * x) * sin(M_PI * x / (double)order) / square(M_PI * x);
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
