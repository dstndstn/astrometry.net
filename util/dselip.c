/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int floatcmp(const void *p1, const void *p2) {
    float f1 = *(float *)p1;
    float f2 = *(float *)p2;
    return (f1 > f2)?1:(f1 < f2)?-1:0;
}

#ifdef SIMPLEXY_REENTRANT

// this is slower, because each call needs to malloc, but it is reentrant
float dselip(unsigned long k, unsigned long n, const float *arr) {
	float* sorted_data = malloc(sizeof(float) * n);
    memcpy(sorted_data, arr, sizeof(float)*n);
    qsort(sorted_data, n, sizeof(float), floatcmp);
	float kth_item = sorted_data[k];
	free(sorted_data);
	return kth_item;
}

void dselip_cleanup() {
}

#else

static int high_water_mark = 0;
static float* past_data = NULL;

float dselip(unsigned long k, unsigned long n, float *arr) {
	if (n > high_water_mark) {
        free(past_data);
		past_data = malloc(sizeof(float) * n);
		high_water_mark = n;
		//printf("dselip watermark=%lu\n",n);
	}
	memcpy(past_data, arr, sizeof(float) * n);
    qsort(past_data, n, sizeof(float), floatcmp);
	return past_data[k];
}

void dselip_cleanup() {
    free(past_data);
    past_data = NULL;
    high_water_mark = 0;
}

#endif
