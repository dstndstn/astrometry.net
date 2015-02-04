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

/*
  Utilities for noise simulations.
*/
#ifndef NOISE_H
#define NOISE_H

/**
   angleArcMin: radius in arcminutes.
 */
void sample_in_circle(const double* center, double angleArcMin,
                      double* point);

void sample_star_in_ring(double* center,
						 double minAngleArcMin,
						 double maxAngleArcMin,
						 double* point);

void sample_field_in_circle(const double* center, double radius,
							double* point);

void add_star_noise(const double* real, double noisestddev, double* noisy);

void add_field_noise(const double* real, double noisestddev, double* noisy);

void compute_star_code(const double* xyz, int dimquads, double* code);

void compute_field_code(const double* pix, int dimquads,
                        double* code, double* scale);

#endif
