/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include "svd.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

void test_nullspace(CuTest* tc) {
    double cov[4] = {-0.93390448957619598, 1.8004204750064117, 0, 0};

    double U[4], V[4], S[2];

	{
		double* pcov[] = { cov, cov+2 };
		double* pU[]   = { U,   U  +2 };
		double* pV[]   = { V,   V  +2 };
		double eps, tol;
		eps = 1e-30;
		tol = 1e-30;
		svd(2, 2, 1, 1, eps, tol, pcov, S, pU, pV);
	}

    CuAssertDblEquals(tc, 0, S[1], 1e-6);
    CuAssertDblEquals(tc, 2.02822373, S[0], 1e-6);

}
