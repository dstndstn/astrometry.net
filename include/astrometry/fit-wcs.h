/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2012 Dustin Lang.

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

#ifndef FIT_WCS_H
#define FIT_WCS_H

#include "astrometry/sip.h"
#include "astrometry/starkd.h"

int fit_sip_coefficients(const double* starxyz,
                         const double* fieldxy,
                         const double* weights,
                         int M,
                         const tan_t* tanin1,
                         int sip_order,
                         int inv_order,
                         sip_t* sipout);

void wcs_shift(tan_t* wcs, double xs, double ys);

/**
 Computed SIP parameters given a set of corresponding points,
 and an initial TAN WCS solution.

 (See fit_sip_wcs_2 if you don't already have a TAN)

 If 'weights' is NULL, uniform weighting will be used.
 */
int fit_sip_wcs(const double* starxyz,
                const double* fieldxy,
                const double* weights,
                int M,
                const tan_t* tanin,
                int sip_order,
                int inv_order,
                int doshift,
                sip_t* sipout
                );

int fit_sip_wcs_2(const double* starxyz,
                  const double* fieldxy,
                  const double* weights,
                  int M,
                  int sip_order,
                  int inv_order,
                  int W, int H,
                  int crpix_center,
                  double* crpix,
                  int doshift,
                  sip_t* sipout
                  );

/**
 Move the tangent point to the given CRPIX, keeping the corresponding
 stars in "starxyz" and "fieldxy" aligned.  It's assumed that "tanin"
 contains a reasonably close WCS solution (eg, from
 fit_wcs).  The output is put in "tanout".  You might want
 to iterate this process, though in my tests the adjustments in the
 second iteration are very minor.
 */
int fit_tan_wcs_move_tangent_point(const double* starxyz,
								   const double* fieldxy,
								   int N,
								   const double* crpix,
								   const tan_t* tanin,
								   tan_t* tanout);

int fit_tan_wcs_move_tangent_point_weighted(const double* starxyz,
											const double* fieldxy,
											const double* weights,
											int N,
											const double* crpix,
											const tan_t* tanin,
											tan_t* tanout);

/*
 Computes a rigid (conformal) TAN WCS projection, based on the correspondence
 between stars and field objects.
 .  starxyz is an array of star positions on the unit sphere.
 .  fieldxy is an array of pixel coordinates.
 .  nobjs   is the number of correspondences; the star at
 .    (starxyz + i*3) corresponds with the field object at (fieldxy + i*2).

 If "p_scale" is specified, the scale of the field will be placed in it.
 It is in units of degrees per pixel.
*/
int fit_tan_wcs(const double* starxyz,
				const double* fieldxy,
				int nobjs,
				// output:
				tan_t* wcstan,
				double* p_scale);

int fit_tan_wcs_weighted(const double* starxyz,
						 const double* fieldxy,
						 const double* weights,
						 int N,
						 // output:
						 tan_t* tan,
						 double* p_scale);

#endif
