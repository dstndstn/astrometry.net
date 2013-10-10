/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg,
  Sam Roweis and Dustin Lang.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "radix.h"

#ifdef SIMPLEXY_REENTRANT

// this is slower, because each call needs to malloc, but it is reentrant
float dselip(unsigned long k, unsigned long n, const float *arr) {
	float* sorted_data = malloc(sizeof(float) * n);
	float* temp_arr = malloc(sizeof(float) * n);
    memcpy(temp_arr, arr, sizeof(float)*n);
	RadixSort11(temp_arr, sorted_data, n);
    free(temp_arr);
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
	float* temp_arr = malloc(sizeof(float) * n);
    memcpy(temp_arr, arr, sizeof(float)*n);
	RadixSort11(temp_arr, past_data, n);
    free(temp_arr);
	return past_data[k];
}

void dselip_cleanup() {
    free(past_data);
    past_data = NULL;
    high_water_mark = 0;
}

#endif
