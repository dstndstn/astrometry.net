/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef NGCIC_ACCURATE_H
#define NGCIC_ACCURATE_H

#include "an-bool.h"

/*
  The accurate NGC/IC positions database can be found here:
    http://www.ngcic.org/corwin/default.htm
*/

struct ngcic_accurate {
  // true: NGC.  false: IC.
  anbool is_ngc;
  // NGC/IC number
  int id;
  // RA,Dec in B2000.0 degrees
  float ra;
  float dec;
};
typedef struct ngcic_accurate ngcic_accurate;

int ngcic_accurate_get_radec(anbool is_ngc, int id, float* ra, float* dec);

int ngcic_accurate_num_entries();

ngcic_accurate* ngcic_accurate_get_entry(int i);

#endif
