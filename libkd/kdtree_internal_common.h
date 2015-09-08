/*
# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#define DIST_SCALE( kd, rd)  ((rd) * (kd)->scale)
#define DIST2_SCALE(kd, rd)  ((rd) * (kd)->scale * (kd)->scale)

#define DIST_INVSCALE( kd, rd)  ((rd) * (kd)->invscale)
#define DIST2_INVSCALE(kd, rd)  ((rd) * (kd)->invscale * (kd)->invscale)

#define POINT_SCALE(   kd, d, p)    (((p) - (kd)->minval[d]) * (kd)->scale)
#define POINT_INVSCALE(kd, d, p)    (((p) * ((kd)->invscale)) + (kd)->minval[d])

