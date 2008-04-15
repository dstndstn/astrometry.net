/*
  This file is part of libkd.
  Copyright 2006-2008 Dustin Lang and Keir Mierle.

  libkd is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, version 2.

  libkd is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with libkd; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#define DIST_SCALE( kd, rd)  ((rd) * (kd)->scale)
#define DIST2_SCALE(kd, rd)  ((rd) * (kd)->scale * (kd)->scale)

#define DIST_INVSCALE( kd, rd)  ((rd) * (kd)->invscale)
#define DIST2_INVSCALE(kd, rd)  ((rd) * (kd)->invscale * (kd)->invscale)

#define POINT_SCALE(   kd, d, p)    (((p) - (kd)->minval[d]) * (kd)->scale)
#define POINT_INVSCALE(kd, d, p)    (((p) * ((kd)->invscale)) + (kd)->minval[d])

