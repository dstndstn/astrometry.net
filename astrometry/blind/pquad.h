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
