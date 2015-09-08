/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

/* 

 Sorting functions of bl.h, to factor out qsort dependency.

 */
#ifndef BL_SORT_H
#define BL_SORT_H

#include "astrometry/bl.h"

void bl_sort(bl* list, int (*compare)(const void* v1, const void* v2));

void  pl_sort(pl* list, int (*compare)(const void* v1, const void* v2));

#endif
