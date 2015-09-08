/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

/*
  Common declarations and syntactic candy.
*/

#ifndef SIMPLEXY_COMMON_H
#define SIMPLEXY_COMMON_H

#include <math.h>

#define PI M_PI

#define FREEVEC(a) {if((a)!=NULL) free((a)); (a)=NULL;}

#endif
