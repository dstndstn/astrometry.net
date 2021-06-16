/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "permutedsort.h"
#include "os-features.h"
#include "ioutils.h"

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
        temparr = malloc((size_t)elemsize * (size_t)Nperm);
        coutput = temparr;
    } else
        coutput = outarray;

    cinput = inarray;
    for (i=0; i<Nperm; i++)
        memcpy(coutput + i * elemsize, cinput + (size_t)perm[i] * (size_t)elemsize, elemsize);

    if (inarray == outarray) {
        memcpy(outarray, temparr, (size_t)elemsize * (size_t)Nperm);
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

#define COMPARE(d1, d2, op1, op2)                       \
    if (d1 op1 d2) return -1;                           \
    if (d1 op2 d2) return 1;                            \
    /* explicitly test for equality, to catch NaNs*/	\
    if (d1 == d2) return 0;                             \
    if (isnan(d1) && isnan(d2)) return 0;               \
    if (isnan(d1)) return 1;                            \
    if (isnan(d2)) return -1;                           \
    assert(0); return 0;

//printf("d1=%g, d2=%g\n", d1, d2);				   

#define INTCOMPARE(i1, i2, op1, op2)            \
    if (i1 op1 i2) return -1;                   \
    if (i1 op2 i2) return 1;                    \
    return 0;

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

int compare_int64_asc(const void* v1, const void* v2) {
    int64_t i1 = *(int64_t*)v1;
    int64_t i2 = *(int64_t*)v2;
    INTCOMPARE(i1, i2, <, >);
}

int compare_int64_desc(const void* v1, const void* v2) {
    int64_t i1 = *(int64_t*)v1;
    int64_t i2 = *(int64_t*)v2;
    INTCOMPARE(i1, i2, >, <);
}

// Versions for use with QSORT_R
int QSORT_COMPARISON_FUNCTION(compare_floats_asc_r,
                              void* thunk, const void* v1, const void* v2) {
    return compare_floats_asc(v1, v2);
}


#undef COMPARE
#undef INTCOMPARE

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

