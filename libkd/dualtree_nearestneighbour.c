/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "os-features.h"
#include "dualtree_nearestneighbour.h"
#include "dualtree.h"
#include "mathutil.h"

struct rs_params {
    kdtree_t* xtree;
    kdtree_t* ytree;

    anbool notself;

    double* node_nearest_d2;

    double d2;
    double* nearest_d2;
    int* nearest_ind;
    int* count_in_range;
};
typedef struct rs_params rs_params;

static anbool rs_within_range(void* params, kdtree_t* searchtree, int searchnode,
                              kdtree_t* querytree, int querynode);
static void rs_handle_result(void* extra, kdtree_t* searchtree, int searchnode,
                             kdtree_t* querytree, int querynode);

void dualtree_nearestneighbour(kdtree_t* xtree, kdtree_t* ytree, double maxdist2,
                               double** nearest_d2, int** nearest_ind,
                               int** count_in_range,
                               int notself) {
    int i, NY, NNY;

    // dual-tree search callback functions
    dualtree_callbacks callbacks;
    rs_params params;

    // These two inputs must be non-NULL (they are essential return values);
    // but they may point to pointers that are NULL (indicating that the caller wants us to
    // allocate and return new arrays).
    assert(nearest_d2);
    assert(nearest_ind);

    memset(&callbacks, 0, sizeof(dualtree_callbacks));
    callbacks.decision = rs_within_range;
    callbacks.decision_extra = &params;
    callbacks.result = rs_handle_result;
    callbacks.result_extra = &params;

    // set search params
    NY = kdtree_n(ytree);
    memset(&params, 0, sizeof(params));
    params.xtree = xtree;
    params.ytree = ytree;
    params.notself = notself;
    params.d2 = maxdist2;

    params.count_in_range = NULL;
    if (count_in_range) {
        if (!(*count_in_range)) {
            *count_in_range = (int*)calloc(NY, sizeof(int));
        }
        params.count_in_range = *count_in_range;
    }

    // were we given a d2 array?
    if (*nearest_d2)
        params.nearest_d2 = *nearest_d2;
    else
        params.nearest_d2 = malloc(NY * sizeof(double));

    if (maxdist2 == 0.0)
        maxdist2 = LARGE_VAL;
    for (i=0; i<NY; i++)
        params.nearest_d2[i] = maxdist2;

    // were we given an ind array?
    if (*nearest_ind)
        params.nearest_ind = *nearest_ind;
    else
        params.nearest_ind = malloc(NY * sizeof(int));
    for (i=0; i<NY; i++)
        params.nearest_ind[i] = -1;

    NNY = kdtree_nnodes(ytree);
    params.node_nearest_d2 = malloc(NNY * sizeof(double));
    for (i=0; i<NNY; i++)
        params.node_nearest_d2[i] = maxdist2;
    
    dualtree_search(xtree, ytree, &callbacks);

    // Return array addresses
    *nearest_d2 = params.nearest_d2;
    *nearest_ind = params.nearest_ind;
    free(params.node_nearest_d2);
}

static anbool rs_within_range(void* vparams,
                              kdtree_t* xtree, int xnode,
                              kdtree_t* ytree, int ynode) {
    rs_params* p = (rs_params*)vparams;
    double maxd2;

    // count-in-range is actually more like rangesearch...
    if (p->count_in_range) {
        if (kdtree_node_node_mindist2_exceeds(xtree, xnode, ytree, ynode, p->d2))
            return FALSE;
        return TRUE;
    }

    if (kdtree_node_node_mindist2_exceeds(xtree, xnode, ytree, ynode,
                                          p->node_nearest_d2[ynode]))
        return FALSE;

    maxd2 = kdtree_node_node_maxdist2(xtree, xnode, ytree, ynode);
    if (maxd2 < p->node_nearest_d2[ynode]) {
        // update this node and its children.
        p->node_nearest_d2[ynode] = maxd2;
        if (!KD_IS_LEAF(ytree, ynode)) {
            int child = KD_CHILD_LEFT(ynode);
            p->node_nearest_d2[child] = MIN(p->node_nearest_d2[child], maxd2);
            child = KD_CHILD_RIGHT(ynode);
            p->node_nearest_d2[child] = MIN(p->node_nearest_d2[child], maxd2);
        }
    }
    return TRUE;
}

/**
 This callback gets called when we've reached a node in the Y tree and
 a node in the X tree (one or both may be leaves), and it's time to
 look at individual data points.
 */
static void rs_handle_result(void* vparams,
                             kdtree_t* xtree, int xnode,
                             kdtree_t* ytree, int ynode) {
    int xl, xr, yl, yr;
    int x, y;
    rs_params* p = (rs_params*)vparams;
    int D = ytree->ndim;
    double checkd2;

    xl = kdtree_left (xtree, xnode);
    xr = kdtree_right(xtree, xnode);
    yl = kdtree_left (ytree, ynode);
    yr = kdtree_right(ytree, ynode);

    for (y=yl; y<=yr; y++) {
        void* py = kdtree_get_data(ytree, y);

        if (p->count_in_range) {
            checkd2 = p->d2;
        } else {
            p->nearest_d2[y] = MIN(p->nearest_d2[y], p->node_nearest_d2[ynode]);
            checkd2 = p->nearest_d2[y];
        }
		
        // check if we can eliminate the whole x node for this y point...
        if (kdtree_node_point_mindist2_exceeds(xtree, xnode, py, checkd2))
            continue;

        for (x=xl; x<=xr; x++) {
            double d2;
            void* px;
            if (p->notself && (y == x))
                continue;
            px = kdtree_get_data(xtree, x);
            d2 = distsq(px, py, D);

            if (p->count_in_range) {
                if (d2 < p->d2) {
                    p->count_in_range[y]++;
                }
            }

            if (d2 > p->nearest_d2[y])
                continue;
            p->nearest_d2[y] = d2;
            p->nearest_ind[y] = x;
        }
    }
}

