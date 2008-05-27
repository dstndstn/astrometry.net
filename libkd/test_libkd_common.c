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

static kdtree_t* build_tree(CuTest* tc, double* data, int N, int D,
                            int Nleaf, int treetype, int treeopts);
static double* random_points_d(int N, int D);

static double* random_points_d(int N, int D) {
    int i;
    double* data = malloc(N * D * sizeof(double));
    for (i=0; i<(N*D); i++) {
        data[i] = rand() / (double)RAND_MAX;
    }
    return data;
}

static kdtree_t* build_tree(CuTest* tc, double* data, int N, int D,
                            int Nleaf, int treetype, int treeopts) {
    kdtree_t* kd;
    kd = kdtree_build(NULL, data, N, D, Nleaf, treetype, treeopts);
    if (!kd)
        return NULL;
    CuAssertIntEquals(tc, kdtree_check(kd), 0);
    return kd;
}
