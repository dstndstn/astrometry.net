/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang

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
#include <assert.h>

#include "solver.h"
#include "index.h"
#include "pquad.h"
#include "log.h"

int compare_ints_ascending(const void* v1, const void* v2) {
    int i1 = *(int*)v1;
    int i2 = *(int*)v2;
    if (i1 > i2) return 1;
    else if (i1 < i2) return -1;
    else return 0;
}

int compare_quad(const void* v1, const void* v2) {
    const uint* u1 = v1;
    const uint* u2 = v2;
    int i;
    for (i=0; i<4; i++) {
        if (u1[i] < u2[i]) return -1;
        if (u1[i] > u2[i]) return 1;
    }
    return 0;
}



bl* quadlist;

void test_try_all_codes(pquad* pq,
                        uint* fieldstars, int dimquad,
                        solver_t* solver, double tol2) {
    uint sorted[dimquad];
    int i;
    fflush(NULL);
    printf("test_try_all_codes: [");
    for (i=0; i<dimquad; i++) {
        printf("%s%i", (i?" ":""), fieldstars[i]);
    }
    printf("]");

    // sort AB and CD...
    memcpy(sorted, fieldstars, dimquad * sizeof(uint));
    qsort(sorted, 2, sizeof(uint), compare_ints_ascending);
    qsort(sorted+2, dimquad-2, sizeof(uint), compare_ints_ascending);

    printf(" -> [");
    for (i=0; i<dimquad; i++) {
        printf("%s%i", (i?" ":""), sorted[i]);
    }
    printf("]\n");
    fflush(NULL);

    bl_append(quadlist, sorted);
    
}

void test1() {
    double field[14];
    int i=0;
    solver_t* solver;
    index_t index;
    uint wanted[][4] = { { 0,1,3,4 },
                         { 0,2,3,4 },
                         { 1,2,3,4 },
                         { 2,5,0,1 },
                         { 2,5,0,3 },
                         { 2,5,0,4 },
                         { 2,5,1,3 },
                         { 2,5,1,4 },
                         { 2,5,3,4 },
                         { 0,1,3,5 },
                         { 0,1,4,5 },
                         { 0,6,4,5 },
                         { 1,6,4,5 },
                         { 2,6,0,1 },
                         { 2,6,0,3 },
                         { 2,6,0,4 },
                         { 2,6,0,5 },
                         { 2,6,1,3 },
                         { 2,6,1,4 },
                         { 2,6,1,5 },
                         { 2,6,3,4 },
                         { 2,6,3,5 },
                         { 2,6,4,5 },
                         { 3,6,0,1 },
                         { 3,6,0,4 },
                         { 3,6,0,5 },
                         { 3,6,1,4 },
                         { 3,6,1,5 },
                         { 3,6,4,5 },
    };


    // star0 A: (0,0)
    field[i++] = 0.0;
    field[i++] = 0.0;

    // star1 B: (2,2)
    field[i++] = 2.0;
    field[i++] = 2.0;

    // star2
    field[i++] = -1.0;
    field[i++] = 3.0;

    // star3
    field[i++] = 0.5;
    field[i++] = 1.5;

    // star4
    field[i++] = 1.0;
    field[i++] = 1.0;

    // star5
    field[i++] = 1.5;
    field[i++] = 0.5;

    // star6
    field[i++] = 3.0;
    field[i++] = -1.0;

    quadlist = bl_new(16, 4*sizeof(uint));

    solver = solver_new();

    memset(&index, 0, sizeof(index_t));
    index.index_scale_lower = 1;
    index.index_scale_upper = 10;

    solver->funits_lower = 0.1;
    solver->funits_upper = 10;

    pl_append(solver->indexes, &index);
    solver->field = field;
    solver->nfield = sizeof(field) / (2 * sizeof(double));

    solver_preprocess_field(solver);

    solver_run(solver);

    solver_free_field(solver);
    solver_free(solver);

    //
    assert(bl_size(quadlist) == (sizeof(wanted) / (4*sizeof(uint))));
    for (i=0; i<bl_size(quadlist); i++) {
        assert(compare_quad(bl_access(quadlist, i), wanted[i]) == 0);
    }

    bl_free(quadlist);
}

char* OPTIONS = "v";

int main(int argc, char** args) {
    int argchar;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 'v':
            log_init(LOG_ALL+1);
            break;
        }

    test1();
    return 0;
}

