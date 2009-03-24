/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang.

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

#ifndef SIP_UTILS_H
#define SIP_UTILS_H

#include "sip.h"

/*
 Finds stars that are inside the bounds of a given field (wcs).

 One of "sip" or "tan" must be non-NULL; if "sip" is non-NULL it is used.

 One of "xyz" or "radec" must be non-NULL.  If both are non-NULL, xyz is used.
 "N" indicates how many elements are in these arrays.  "radec" are in degrees.

 If "inds" is non-NULL, the indices of stars that are inside the field are
 put there; otherwise a new int array is allocated and returned; it should
 be free()'d.

 The pixel (xy) positions are placed into a newly-allocated array at "xy",
 unless "xy" is NULL.

 The number of good stars is placed in Ngood, which must be non-NULL.
 */
int* sip_filter_stars_in_field(const sip_t* sip, const tan_t* tan,
							   const double* xyz, const double* radec,
							   int N,
							   double** xy, int* inds, int* Ngood);

void sip_get_radec_bounds(const sip_t* wcs, int stepsize,
                          double* pramin, double* pramax,
                          double* pdecmin, double* pdecmax);

// sets RA,Dec in degrees.
void sip_get_radec_center(const sip_t* wcs,
                          double* p_ra, double* p_dec);

// RA hours:minutes:seconds, Dec degrees:minutes:seconds
void sip_get_radec_center_hms(const sip_t* wcs,
                              int* rah, int* ram, double* ras,
                              int* decd, int* decm, double* decs);

// Writes RA, Dec H:M:S and D:M:S strings.
void sip_get_radec_center_hms_string(const sip_t* wcs,
                                     char* rastr, char* decstr);

void sip_get_field_size(const sip_t* wcs,
                        double* pw, double* ph,
                        char** units);

#endif
