/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <stdlib.h>
#include <math.h>
#include <sys/param.h>

#include "resample.h"

#include "mathutil.h"
#include "errors.h"
#include "log.h"

double lanczos(double x, int order) {
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
