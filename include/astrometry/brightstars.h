/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef BRIGHTSTARS_H
#define BRIGHTSTARS_H

struct brightstar {
	// Don't change the order of these fields - the included datafile depends on this order!
	char* name;
	char* common_name;
	double ra;
	double dec;
	double Vmag;
};
typedef struct brightstar brightstar_t;

int bright_stars_n();
const brightstar_t* bright_stars_get(int starindex);

#endif
