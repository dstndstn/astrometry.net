/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

InlineDefine int dimquad2dimcode(int dimquad) {
    return 2 * (dimquad - 2);
}

InlineDefine double rad2deg(double x) {
    return x * 180.0/M_PI;
}

InlineDefine double deg2rad(double x) {
    return x * M_PI/180.0;
}

InlineDefine double deg2arcmin(double x) {
    return x * 60.0;
}

InlineDefine double arcmin2deg(double x) {
    return x / 60.0;
}

InlineDefine double arcmin2arcsec(double x) {
    return x * 60.0;
}

InlineDefine double arcsec2arcmin(double x) {
    return x / 60.0;
}

InlineDefine double rad2arcmin(double x) {
    return deg2arcmin(rad2deg(x));
}

InlineDefine double rad2arcsec(double x) {
    return deg2arcsec(rad2deg(x));
}

InlineDefine double deg2arcsec(double x) {
    return arcmin2arcsec(deg2arcmin(x));
}

InlineDefine double arcmin2rad(double x) {
    return deg2rad(arcmin2deg(x));
}

InlineDefine double arcsec2rad(double x) {
    return deg2rad(arcsec2deg(x));
}

InlineDefine double arcsec2deg(double x) {
    return arcmin2deg(arcsec2arcmin(x));
}

