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

#include <assert.h>

#include "brightstars.h"

static brightstar_t bs[] =
#include "brightstars-data.c"
;

int bright_stars_n() {
	return sizeof(bs) / sizeof(brightstar_t);
}

const brightstar_t* bright_stars_get(int starindex) {
	assert(starindex >= 0);
	assert(starindex < bright_stars_n());
	return bs + starindex;
}

