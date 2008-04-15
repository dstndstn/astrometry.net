/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef BLIND_WCS_H
#define BLIND_WCS_H

#include "qfits.h"
#include "matchobj.h"
#include "sip.h"
#include "starkd.h"

/*
  Computes a rigid TAN WCS projection, based on the correspondence
  between stars and field objects.
  . starxyz is an array of star positions on the unit sphere.
  . fieldxy is an array of pixel coordinates.
  . nobjs   is the number of correspondences; the star at
  .    (starxyz + i*3) corresponds with the field object at (fieldxy + i*2).
  
  If "p_scale" is specified, the scale of the field will be placed in it.
  It is in units of degrees per pixel, and equals sqrt(abs(det(CD))).
*/
void blind_wcs_compute(double* starxyz,
					   double* fieldxy,
					   int nobjs,
					   // output:
					   tan_t* wcstan,
					   double* p_scale);

qfits_header* blind_wcs_get_header(tan_t* wcstan);

qfits_header* blind_wcs_get_sip_header(sip_t* wcstan);

#endif
