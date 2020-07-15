/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
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
