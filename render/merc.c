/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Keir Mierle and Dustin Lang.

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

#include <math.h>

#include "starutil.h"
#include "merc.h"

// Converts from RA in radians to Mercator X coordinate in [0, 1].
double ra2merc(double ra) {
	return 1.0 - (ra / (2.0 * M_PI));
}

// Converts from Mercator X coordinate [0, 1] to RA in radians.
double merc2ra(double x) {
    // here's the flip!
	return (1.0 - x) * (2.0 * M_PI);
}

// Converts from Dec in radians to Mercator Y coordinate in [0, 1].
double dec2merc(double dec) {
	return 0.5 + (asinh(tan(dec)) / (2.0 * M_PI));
}

// Converts from Mercator Y coordinate [0, 1] to DEC in radians.
double merc2dec(double y) {
	return atan(sinh((y - 0.5) * (2.0 * M_PI)));
}

// Converts from RA in degrees to Mercator X coordinate in [0, 1].
double radeg2merc(double ra) {
	return ra2merc(deg2rad(ra));
}

// Converts from Mercator X coordinate [0, 1] to RA in degrees.
double merc2radeg(double x) {
	return rad2deg(merc2ra(x));
}

// Converts from Dec in degrees to Mercator X coordinate in [0, 1].
double decdeg2merc(double ra) {
	return dec2merc(deg2rad(ra));
}

// Converts from Mercator Y coordinate [0, 1] to DEC in degrees.
double merc2decdeg(double y) {
	return rad2deg(merc2dec(y));
}


