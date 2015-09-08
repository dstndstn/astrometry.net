/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/
#ifndef JPL_H
#define JPL_H

#include "astrometry/bl.h"

struct orbital_elements {
	double a;
	double e;
	double I;
	double Omega;
	double pomega;
	double M;
	double mjd;
};
typedef struct orbital_elements orbital_elements_t;

bl* jpl_parse_orbital_elements(const char* str, bl* lst);


#endif


