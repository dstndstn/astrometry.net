/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>

#include "dimage.h"
#include "cutest.h"
#include "simplexy-common.h"

extern int initial_max_groups;

static int old_dfind(int *image, int nx, int ny, int *object) {
    int i, ip, j, jp, k, kp, l, ist, ind, jst, jnd, igroup, minearly, checkearly, tmpearly;
    int ngroups;

    int* mapgroup = (int *) malloc((size_t) nx * ny * sizeof(int));
    int* matches = (int *) malloc((size_t) nx * ny * 9 * sizeof(int));
    int* nmatches = (int *) malloc((size_t) nx * ny * sizeof(int));

    if (!mapgroup || !matches || !nmatches) {
        fprintf(stderr, "Failed to allocate memory in dfind.c\n");
        exit(-1);
    }

    for (k = 0;k < nx*ny;k++)
        object[k] = -1;
    for (k = 0;k < nx*ny;k++)
        mapgroup[k] = -1;
    for (k = 0;k < nx*ny;k++)
        nmatches[k] = 0;
    for (k = 0;k < nx*ny*9;k++)
        matches[k] = -1;

    /* find matches */
    for (j = 0;j < ny;j++) {
        jst = j - 1;
        jnd = j + 1;
        if (jst < 0)
            jst = 0;
        if (jnd > ny - 1)
            jnd = ny - 1;
        for (i = 0;i < nx;i++) {
            ist = i - 1;
            ind = i + 1;
            if (ist < 0)
                ist = 0;
            if (ind > nx - 1)
                ind = nx - 1;
            k = i + j * nx;
            if (image[k]) {
                for (jp = jst;jp <= jnd;jp++)
                    for (ip = ist;ip <= ind;ip++) {
                        kp = ip + jp * nx;
                        if (image[kp]) {
                            matches[9*k + nmatches[k]] = kp;
                            nmatches[k]++;
                        }
                    }
            } /* end if */
        }
    }

    /* group pixels on matches */
    igroup = 0;
    for (k = 0;k < nx*ny;k++) {
        if (image[k]) {
            minearly = igroup;
            for (l = 0;l < nmatches[k];l++) {
                kp = matches[9 * k + l];
                checkearly = object[kp];
                if (checkearly >= 0) {
                    while (mapgroup[checkearly] != checkearly) {
                        checkearly = mapgroup[checkearly];
                    }
                    if (checkearly < minearly)
                        minearly = checkearly;
                }
            }

            if (minearly == igroup) {
                mapgroup[igroup] = igroup;
                for (l = 0;l < nmatches[k];l++) {
                    kp = matches[9 * k + l];
                    object[kp] = igroup;
                }
                igroup++;
            } else {
                for (l = 0;l < nmatches[k];l++) {
                    kp = matches[9 * k + l];
                    checkearly = object[kp];
                    if (checkearly >= 0) {
                        while (mapgroup[checkearly] != checkearly) {
                            tmpearly = mapgroup[checkearly];
                            mapgroup[checkearly] = minearly;
                            checkearly = tmpearly;
                        }
                        mapgroup[checkearly] = minearly;
                    }
                }
                for (l = 0;l < nmatches[k];l++) {
                    kp = matches[9 * k + l];
                    object[kp] = minearly;
                }
            }
        }
    }

    ngroups = 0;
    for (i = 0;i < nx*ny;i++) {
        if (mapgroup[i] >= 0) {
            if (mapgroup[i] == i) {
                mapgroup[i] = ngroups;
                ngroups++;
            } else {
                mapgroup[i] = mapgroup[mapgroup[i]];
            }
        }
    }

    if (ngroups == 0)
        goto bail;

    for (i = 0;i < nx*ny;i++)
        if (object[i] >= 0)
            object[i] = mapgroup[object[i]];

    for (i = 0;i < nx*ny;i++)
        mapgroup[i] = -1;
    igroup = 0;
    for (k = 0;k < nx*ny;k++) {
        if (image[k] > 0 && mapgroup[object[k]] == -1) {
            mapgroup[object[k]] = igroup;
            igroup++;
        }
    }

    for (i = 0;i < nx*ny;i++)
        if (image[i] > 0)
            object[i] = mapgroup[object[i]];
        else
            object[i] = -1;

 bail:
    FREEVEC(matches);
    FREEVEC(nmatches);
    FREEVEC(mapgroup);

    return (1);
}

int compare_inputs(int *test_data, int nx, int ny) {
    int *test_outs_keir = calloc(nx*ny, sizeof(int));
    int *test_outs_blanton = calloc(nx*ny, sizeof(int));
    int *test_outs_u8 = calloc(nx*ny, sizeof(int));
    int fail = 0;
    int ix, iy, i;
    unsigned char* u8img;

    dfind2(test_data, nx,ny,test_outs_keir, NULL);
    old_dfind(test_data, nx,ny,test_outs_blanton);

    u8img = malloc(nx * ny);
    for (i=0; i<(nx*ny); i++)
        u8img[i] = test_data[i];
    dfind2_u8(u8img, nx, ny, test_outs_u8, NULL);

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
    dimage_label_t equivs[] = {0, 0, 1, 2};
    /* 0  1  2  3 */
    dimage_label_t minlabel = collapsing_find_minlabel(3, equivs);
    CuAssertIntEquals(tc, minlabel, 0);
    CuAssertIntEquals(tc, equivs[0], 0);
    CuAssertIntEquals(tc, equivs[1], 0);
    CuAssertIntEquals(tc, equivs[2], 0);
    CuAssertIntEquals(tc, equivs[3], 0);
}
