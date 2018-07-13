/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include "kdtree.h"

void ex1() {
    double mydata[] = { 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8 };
    int D = 2;
    int N = sizeof(mydata) / (D * sizeof(double));
    printf("N %i, D %i\n", N, D);
    kdtree_t* kd = kdtree_build(NULL, mydata, N, D, 4, KDTT_DOUBLE, KD_BUILD_BBOX);
    kdtree_print(kd);
    kdtree_free(kd);
}

int main() {
    ex1();
}
