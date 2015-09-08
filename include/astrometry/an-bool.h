/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef AN_BOOL_H
#define AN_BOOL_H

#include <stdint.h>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

// This helps unconfuse SWIG; it doesn't seem to like uint8_t
typedef unsigned char anbool;

#endif
