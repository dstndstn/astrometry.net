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

#ifndef MERC_H
#define MERC_H

// Converts from RA in radians to Mercator X coordinate in [0, 1].
double ra2merc(double ra);

// Converts from RA in degrees to Mercator X coordinate in [0, 1].
double radeg2merc(double ra);

// Converts from Mercator X coordinate [0, 1] to RA in radians.
double merc2ra(double x);

// Converts from Mercator X coordinate [0, 1] to RA in degrees.
double merc2radeg(double x);

// Converts from DEC in radians to Mercator Y coordinate in [0, 1].
double dec2merc(double dec);

// Converts from DEC in degrees to Mercator Y coordinate in [0, 1].
double decdeg2merc(double dec);

// Converts from Mercator Y coordinate [0, 1] to DEC in radians.
double merc2dec(double y);

// Converts from Mercator Y coordinate [0, 1] to DEC in degrees.
double merc2decdeg(double y);

#endif
