/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"

#include "jpl.h"
#include "bl.h"
#include "starutil.h"

void test_jpl_1(CuTest* tc) {
	bl* lst;
	orbital_elements_t* orb;
	char* s = "*******************************************************************************
$$SOE
2451544.500000000 = A.D. 2000-Jan-01 00:00:00.0000 (CT)
 EC= 5.572196627498378E-02 QR= 9.048090920148466E+00 IN= 2.485240160067824E+00
 OM= 1.136428218226837E+02 W = 3.360118627136669E+02 Tp=  2452738.076490710024
 N = 3.323384124604674E-02 MA= 3.203328683927277E+02 TA= 3.160307045699099E+02
 A = 9.582019910444478E+00 AD= 1.011594890074049E+01 PR= 1.083233193944510E+04
2451551.500000000 = A.D. 2000-Jan-08 00:00:00.0000 (CT)
 EC= 5.573960302454895E-02 QR= 9.047888350190947E+00 IN= 2.485243118146288E+00
 OM= 1.136426544365385E+02 W = 3.360396713123293E+02 Tp=  2452738.795322154649
 N = 3.323402622894304E-02 MA= 3.205413961220120E+02 TA= 3.162559512605895E+02
 A = 9.581984354286307E+00 AD= 1.011608035838167E+01 PR= 1.083227164593380E+04
2451558.500000000 = A.D. 2000-Jan-15 00:00:00.0000 (CT)
 EC= 5.575752058883327E-02 QR= 9.047688680528223E+00 IN= 2.485246705983557E+00
 OM= 1.136424539461901E+02 W = 3.360670860115268E+02 Tp=  2452739.502766492777
 N = 3.323418041469598E-02 MA= 3.207503409881191E+02 TA= 3.164817069736243E+02
 A = 9.581954717998279E+00 AD= 1.011622075546834E+01 PR= 1.083222139098727E+04
";

	lst = jpl_parse_orbital_elements(s, NULL);
	CuAssertPtrNotNull(tc, lst);
	CuAssertIntEquals(tc, 3, bl_size(lst));
	orb = bl_access(lst, 0);
	CuAssertDblEquals(tc, 2451544.500000000, mjdtojd(orb->mjd), 1e-6);
	CuAssertDblEquals(tc, 5.572196627498378E-02, orb->e, fabs(orb->e)*1e-10);

	/*
	 EC= 5.572196627498378E-02 QR= 9.048090920148466E+00 IN= 2.485240160067824E+00
	 OM= 1.136428218226837E+02 W = 3.360118627136669E+02 Tp=  2452738.076490710024
	 N = 3.323384124604674E-02 MA= 3.203328683927277E+02 TA= 3.160307045699099E+02
	 A = 9.582019910444478E+00 AD= 1.011594890074049E+01 PR= 1.083233193944510E+04
	 */

}



