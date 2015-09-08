/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef PQUAD_H
#define PQUAD_H

/**
 This file is just required for testing purposes (of solver.c)
 */

struct potential_quad
{
	anbool scale_ok;
	int fieldA, fieldB;
	// distance-squared between A and B, in pixels^2.
	double scale;
	double costheta, sintheta;
	// (field pixel noise / quad scale in pixels)^2
	double rel_field_noise2;
	anbool* inbox;
	int ninbox;
	double* xy;
};
typedef struct potential_quad pquad;

#endif
