/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <stdlib.h>

#include "cutest.h"
#include "solverutils.h"
#include "bl.h"

static void assertListEquals(CuTest* tc, int* expect, int N, il* lst) {
    int i;
    CuAssertIntEquals(tc, N, il_size(lst));
    for (i=0; i<N; i++)
        CuAssertIntEquals(tc, expect[i], il_get(lst, i));
}

void test_depths(CuTest* tc) {
    il* lst = il_new(4);
    int rtn;

    {
        int e[] = { 1, 10, 11, 20, 21, 30, 31, 50, 51, 90};
        int N = sizeof(e)/sizeof(int);
        rtn = parse_depth_string(lst, "10, 20, 30, 50, 90");
        CuAssertIntEquals(tc, 0, rtn);
        assertListEquals(tc, e, N, lst);
        il_remove_all(lst);
    }
    {
        int e[] = { 1, 20, 42, 42, 90, 0 };
        int N = sizeof(e)/sizeof(int);
        rtn = parse_depth_string(lst, "-20 42-42 90-");
        CuAssertIntEquals(tc, 0, rtn);
        assertListEquals(tc, e, N, lst);
        il_remove_all(lst);
    }
    {
        int e[] = { 5, 10 };
        int N = sizeof(e)/sizeof(int);
        rtn = parse_depth_string(lst, "5-10   ");
        CuAssertIntEquals(tc, 0, rtn);
        assertListEquals(tc, e, N, lst);
        il_remove_all(lst);
    }

    il_free(lst);
}
