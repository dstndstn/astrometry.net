/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Michael Blanton, Keir Mierle, David W. Hogg, Sam Roweis
  and Dustin Lang.

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

#include "cutest.h"

extern int initial_max_groups;

int dfind(int *image, int nx, int ny, int *object);
int dfind2(int *image, int nx, int ny, int *object);
int dfind2_u8(unsigned char *image, int nx, int ny, int *object);

int compare_inputs(int *test_data, int nx, int ny) {
	int *test_outs_keir = calloc(nx*ny, sizeof(int));
	int *test_outs_blanton = calloc(nx*ny, sizeof(int));
	int *test_outs_u8 = calloc(nx*ny, sizeof(int));
	int fail = 0;
	int ix, iy, i;
	unsigned char* u8img;

	dfind2(test_data, nx,ny,test_outs_keir);
	dfind(test_data, nx,ny,test_outs_blanton);

	u8img = malloc(nx * ny);
	for (i=0; i<(nx*ny); i++)
		u8img[i] = test_data[i];
	dfind2_u8(u8img, nx, ny, test_outs_u8);

	for(iy=0; iy<ny; iy++) {
		for (ix=0; ix<nx; ix++) {
			if (!(test_outs_keir[nx*iy+ix] == test_outs_blanton[nx*iy+ix])) {
				printf("failure -- k%d != b%d\n",
						test_outs_keir[nx*iy+ix], test_outs_blanton[nx*iy+ix]);
				fail++;
			}
			if (!(test_outs_keir[nx*iy+ix] == test_outs_u8[nx*iy+ix])) {
				printf("failure -- k:%d != u8:%d\n",
						test_outs_keir[nx*iy+ix], test_outs_u8[nx*iy+ix]);
				fail++;
			}
		}
	}

    free(u8img);
    free(test_outs_keir);
    free(test_outs_blanton);
    free(test_outs_u8);

	return fail;
}


void test_empty(CuTest* tc) {
	initial_max_groups = 1;
	int test_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	CuAssertIntEquals(tc, compare_inputs(test_data, 11, 9), 0);
}

void test_medium(CuTest* tc) {
	initial_max_groups = 1;
	int test_data[] = {1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
	                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
	                   0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
	                   0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
	                   0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
	                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
	                   0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
	                   0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0};
	CuAssertIntEquals(tc,compare_inputs(test_data, 11, 9),0);
}

void test_tricky(CuTest* tc) {
	initial_max_groups = 1;
	int test_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
	                   1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
	                   1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
	                   0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1,
	                   1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
	                   1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
	                   0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0};
	CuAssertIntEquals(tc,compare_inputs(test_data, 11, 9),0);
}

void test_nasty(CuTest* tc) {
	initial_max_groups = 1;
	int test_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
	                   1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
	                   1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
	                   0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
	                   1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
	                   1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
	                   0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0};
	CuAssertIntEquals(tc,compare_inputs(test_data, 11, 9),0);
}

void test_very_nasty(CuTest* tc) {
	initial_max_groups = 1;
	int test_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                   1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
	                   1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
	                   1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
	                   0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
	                   1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
	                   0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
	                   0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0};
	CuAssertIntEquals(tc, compare_inputs(test_data, 11, 9),0);
}

void test_collapsing_find_simple(CuTest* tc) {
	short int equivs[] = {0, 0, 1, 2};
	             /* 0  1  2  3 */
	short int minlabel = collapsing_find_minlabel(3, equivs);
	CuAssertIntEquals(tc, minlabel, 0);
	CuAssertIntEquals(tc, equivs[0], 0);
	CuAssertIntEquals(tc, equivs[1], 0);
	CuAssertIntEquals(tc, equivs[2], 0);
	CuAssertIntEquals(tc, equivs[3], 0);
}
