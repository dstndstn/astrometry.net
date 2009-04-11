/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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

#include <unistd.h>

#include "starutil.h"
#include "cutest.h"

void test_atora(CuTest* tc) {
    CuAssertDblEquals(tc, 0.0, atora("00:00:00.0"), 1e-6);
    CuAssertDblEquals(tc, 15.0, atora("01:00:00.0"), 1e-6);
    CuAssertDblEquals(tc, 0.25, atora("00:01:00.0"), 1e-6);
    CuAssertDblEquals(tc, 0.25/60.0, atora("00:00:01.0"), 1e-6);
    CuAssertDblEquals(tc, 0.25/60.0/10.0, atora("00:00:00.1"), 1e-6);
}
void test_atodec(CuTest* tc) {
    CuAssertDblEquals(tc, 0.0, atodec("00:00:00.0"), 1e-6);
    CuAssertDblEquals(tc, 1.0, atodec("01:00:00.0"), 1e-6);
    CuAssertDblEquals(tc, 1.0/60.0, atodec("00:01:00.0"), 1e-6);
    CuAssertDblEquals(tc, 1.0/3600.0, atodec("00:00:01.0"), 1e-6);
    CuAssertDblEquals(tc, 1.0/36000.0, atodec("00:00:00.1"), 1e-6);
    CuAssertDblEquals(tc, -1.0, atodec("-01:00:00.0"), 1e-6);
    CuAssertDblEquals(tc, -(1.0 + 1.0/60.0), atodec("-01:01:00.0"), 1e-6);
}

