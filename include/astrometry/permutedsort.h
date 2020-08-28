/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef PERMUTED_SORT_H
#define PERMUTED_SORT_H

// for QSORT_COMPARISON_FUNCTION
#include "ioutils.h"

/*
 Computes the permutation array that will cause the "realarray" to be
 sorted according to the "compare" function.

 Ie, the first element in the sorted array will be at
 (char*)realarray + perm[0] * array_stride

 The "stride" parameter gives the number of bytes between successive entries
 in "realarray".

 If "perm" is NULL, a new permutation array will be allocated and returned.
 Otherwise, the permutation array will be placed in "perm".

 Note that if you pass in a non-NULL "perm" array, its existing values will
 be used!  You probably want to initialize it with "permutation_init()" to
 set it to the identity permutation.
 */
int* permuted_sort(const void* realarray, int array_stride,
                   int (*compare)(const void*, const void*),
                   int* perm, int Nperm);

int* permutation_init(int* perm, int Nperm);

/**
 Applies a permutation array to a data vector.

 Copies "inarray" into "outarray" according to the given "perm"utation.

 This also works when "inarray" == "outarray".
 */
void permutation_apply(const int* perm, int Nperm, const void* inarray,
					   void* outarray, int elemsize);

/*
  Some sort functions that might come in handy:
 */
int compare_doubles_asc(const void* v1, const void* v2);
int compare_doubles_desc(const void* v1, const void* v2);

int compare_floats_asc(const void* v1, const void* v2);
int compare_floats_desc(const void* v1, const void* v2);

int compare_int64_asc(const void* v1, const void* v2);
int compare_int64_desc(const void* v1, const void* v2);

int compare_ints_asc(const void* v1, const void* v2);
int compare_ints_desc(const void* v1, const void* v2);

int compare_uchars_asc(const void* v1, const void* v2);
int compare_uchars_desc(const void* v1, const void* v2);

/*
 Versions for use with QSORT_R directly (not with permuted_sort).
 */
int QSORT_COMPARISON_FUNCTION(compare_floats_asc_r,
                              void* thunk, const void* v1, const void* v2);


#endif
