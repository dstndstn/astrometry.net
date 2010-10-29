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

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "permutedsort.h"
#include "os-features.h" // for qsort_r

int* permutation_init(int* perm, int N) {
	int i;
	if (!N)
		return perm;
    if (!perm)
        perm = malloc(sizeof(int) * N);
	for (i=0; i<N; i++)
		perm[i] = i;
	return perm;
}

void permutation_apply(const int* perm, int Nperm, const void* inarray,
					   void* outarray, int elemsize) {
	void* temparr = NULL;
	int i;
	const char* cinput;
	char* coutput;

	if (inarray == outarray) {
		temparr = malloc(elemsize * Nperm);
		coutput = temparr;
	} else
		coutput = outarray;

	cinput = inarray;
	for (i=0; i<Nperm; i++)
		memcpy(coutput + i * elemsize, cinput + perm[i] * elemsize, elemsize);

	if (inarray == outarray) {
		memcpy(outarray, temparr, elemsize * Nperm);
		free(temparr);
	}
}

struct permuted_sort_t {
    int (*compare)(const void*, const void*);
    const void* data_array;
    int data_array_stride;
};
typedef struct permuted_sort_t permsort_t;

// This is the comparison function we use.
static int QSORT_COMPARISON_FUNCTION(compare_permuted, void* user, const void* v1, const void* v2) {
    permsort_t* ps = user;
	off_t i1 = *(int*)v1;
	off_t i2 = *(int*)v2;
	const void *val1, *val2;
    const char* darray = ps->data_array;
	val1 = darray + i1 * ps->data_array_stride;
	val2 = darray + i2 * ps->data_array_stride;
	return ps->compare(val1, val2);
}

int* permuted_sort(const void* realarray, int array_stride,
                   int (*compare)(const void*, const void*),
                   int* perm, int N) {
    permsort_t ps;
	if (!perm)
		perm = permutation_init(perm, N);

    ps.compare = compare;
    ps.data_array = realarray;
    ps.data_array_stride = array_stride;

    QSORT_R(perm, N, sizeof(int), &ps, compare_permuted);

    return perm;
}

#define COMPARE(d1, d2, op1, op2)						\
	if (d1 op1 d2) return -1;							\
	if (d1 op2 d2) return 1;							\
	/* explicitly test for equality, to catch NaNs*/	\
	if (d1 == d2) return 0;								\
	if (isnan(d1) && isnan(d2)) return 0;				\
	if (isnan(d1)) return 1;							\
	if (isnan(d2)) return -1;							\
	assert(0); return 0;

	//printf("d1=%g, d2=%g\n", d1, d2);				   

int compare_doubles_asc(const void* v1, const void* v2) {
	const double d1 = *(double*)v1;
	const double d2 = *(double*)v2;
	COMPARE(d1, d2, <, >);
}

int compare_doubles_desc(const void* v1, const void* v2) {
    // (note that v1,v2 are flipped)
	const double d1 = *(double*)v1;
	const double d2 = *(double*)v2;
	COMPARE(d1, d2, >, <);
}

int compare_floats_asc(const void* v1, const void* v2) {
	float f1 = *(float*)v1;
	float f2 = *(float*)v2;
	COMPARE(f1, f2, <, >);
}

int compare_floats_desc(const void* v1, const void* v2) {
	float f1 = *(float*)v1;
	float f2 = *(float*)v2;
	COMPARE(f1, f2, >, <);
}

#undef COMPARE

int compare_ints_asc(const void* v1, const void* v2) {
	const int d1 = *(int*)v1;
	const int d2 = *(int*)v2;
	if (d1 < d2)
		return -1;
	if (d1 > d2)
		return 1;
	return 0;
}

int compare_ints_desc(const void* v1, const void* v2) {
    return compare_ints_asc(v2, v1);
}

int compare_uchars_asc(const void* v1, const void* v2) {
	const unsigned char d1 = *(unsigned char*)v1;
	const unsigned char d2 = *(unsigned char*)v2;
	if (d1 < d2)
		return -1;
	if (d1 > d2)
		return 1;
	return 0;
}

int compare_uchars_desc(const void* v1, const void* v2) {
    return compare_ints_desc(v2, v1);
}

