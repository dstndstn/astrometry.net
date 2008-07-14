/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>

#include "cutest.h"

#include "dualtree_nearestneighbour.h"
#include "mathutil.h"

void test_nn_1(CuTest* tc) {
    int NX = 100;
    int NY = 100;
    int D = 2;
    int Nleaf = 5;

    int i, j;

    kdtree_t* xkd;
    kdtree_t* ykd;

    double* xdata;
    double* ydata;

    double* nearest_d2 = NULL;
    int* nearest_ind = NULL;

    double* true_nearest_d2;
    int* true_nearest_ind;

    srand(0);

    xdata = malloc(NX * D * sizeof(double));
    ydata = malloc(NY * D * sizeof(double));

    for (i=0; i<(NX*D); i++)
        xdata[i] = rand() / (double)RAND_MAX;
    for (i=0; i<(NY*D); i++)
        ydata[i] = rand() / (double)RAND_MAX;

    true_nearest_d2  = malloc(NY * sizeof(double));
    true_nearest_ind = malloc(NY * sizeof(int));
    for (j=0; j<NY; j++) {
        int ind = -1;
        double bestd2 = HUGE_VAL;
        for (i=0; i<NX; i++) {
            double d2 = distsq(xdata + i*D, ydata + j*D, D);
            if (d2 < bestd2) {
                bestd2 = d2;
                ind = i;
            }
        }
        true_nearest_d2[j] = bestd2;
        true_nearest_ind[j] = ind;
    }

    xkd = kdtree_build(NULL, xdata, NX, D, Nleaf, KDTT_DOUBLE, KD_BUILD_BBOX);
    ykd = kdtree_build(NULL, ydata, NY, D, Nleaf, KDTT_DOUBLE, KD_BUILD_BBOX);

    dualtree_nearestneighbour(xkd, ykd, 1000.0, &nearest_d2, &nearest_ind);

    for (j=0; j<NY; j++) {
        int jj = kdtree_permute(ykd, j);
        /*
         if ((true_nearest_ind[jj] != kdtree_permute(xkd, nearest_ind[j])) ||
         (fabs(true_nearest_d2[jj] - nearest_d2[j]) > 1e-6)) {
         printf("y point %i: nearest %i / %i (%i)\n", j, true_nearest_ind[j],
         nearest_ind[jj], kdtree_permute(xkd, nearest_ind[jj]));
         printf("  dist %g / %g\n", true_nearest_d2[j], nearest_d2[jj]);
         }
         */
        CuAssertIntEquals(tc, true_nearest_ind[jj], kdtree_permute(xkd, nearest_ind[j]));
        CuAssertDblEquals(tc, true_nearest_d2[jj],  nearest_d2[j], 1e-6);
    }

    free(nearest_d2);
    free(nearest_ind);
    free(true_nearest_d2);
    free(true_nearest_ind);

    kdtree_free(xkd);
    kdtree_free(ykd);
    free(xdata);
    free(ydata);
}

