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

#include <stdlib.h>

#include "ngcic-accurate.h"

static ngcic_accurate ngcic_acc[] = {
  #include "ngcic-accurate-entries.c"
};

int ngcic_accurate_get_radec(anbool is_ngc, int id, float* ra, float* dec) {
  int i, N;
  N = sizeof(ngcic_acc) / sizeof(ngcic_accurate);
  for (i=0; i<N; i++) {
    if ((ngcic_acc[i].is_ngc != is_ngc) ||
	(ngcic_acc[i].id != id))
      continue;
    *ra = ngcic_acc[i].ra;
    *dec = ngcic_acc[i].dec;
    return 0;
  }
  return -1;
}

int ngcic_accurate_num_entries() {
  return sizeof(ngcic_acc) / sizeof(ngcic_accurate);
}

ngcic_accurate* ngcic_accurate_get_entry(int i) {
  int N = sizeof(ngcic_acc) / sizeof(ngcic_accurate);
  if (i < 0 || i >= N)
    return NULL;
  return ngcic_acc + i;
}
