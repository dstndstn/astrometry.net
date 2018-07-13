/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>

#include "dualtree_rangesearch.h"
#include "dualtree.h"
#include "mathutil.h"

double RANGESEARCH_NO_LIMIT = 1.12345e308;

struct rs_params {
    kdtree_t* xtree;
    kdtree_t* ytree;

    anbool notself;

    // radius-squared of the search range.
    double mindistsq;
    double maxdistsq;

    // are we using the min/max limit?
    anbool usemin;
    anbool usemax;

    // for "search"
    result_callback user_callback;
    void* user_callback_param;

    progress_callback user_progress;
    void* user_progress_param;
    int ydone;

    double (*distsquared)(void* px, void* py, int D);

    // for "count"
    int* counts;
};
typedef struct rs_params rs_params;

static anbool rs_within_range(void* params, kdtree_t* searchtree, int searchnode,
                              kdtree_t* querytree, int querynode);
static void rs_handle_result(void* extra, kdtree_t* searchtree, int searchnode,
                             kdtree_t* querytree, int querynode);
static void rs_start_results(void* extra, kdtree_t* querytree, int querynode);

static double mydistsq(void* v1, void* v2, int D) {
    return distsq((double*)v1, (double*)v2, D);
}

void dualtree_rangesearch(kdtree_t* xtree, kdtree_t* ytree,
                          double mindist, double maxdist,
                          int notself,
                          dist2_function distsquared,
                          result_callback callback,
                          void* param,
                          progress_callback progress,
                          void* progress_param) {
    // dual-tree search callback functions
    dualtree_callbacks callbacks;
    rs_params params;

    memset(&callbacks, 0, sizeof(dualtree_callbacks));
    callbacks.decision = rs_within_range;
    callbacks.decision_extra = &params;
    callbacks.result = rs_handle_result;
    callbacks.result_extra = &params;

    // set search params
    memset(&params, 0, sizeof(params));
    if ((mindist == RANGESEARCH_NO_LIMIT) || (mindist == 0.0)) {
        params.usemin = FALSE;
    } else {
        params.usemin = TRUE;
        params.mindistsq = mindist * mindist;
    }

    if (maxdist == RANGESEARCH_NO_LIMIT) {
        params.usemax = FALSE;
    } else {
        double d = maxdist;
        /*
         printf("Original distance %.16g\n", d);
         d = kdtree_get_conservative_query_radius(xtree, d);
         printf("Conservative distance in tree 1: %.16g\n", d);
         d = kdtree_get_conservative_query_radius(ytree, d);
         printf("Conservative distance in tree 2: %.16g\n", d);
         */
        params.usemax = TRUE;
        params.maxdistsq = d*d;
    }
    params.notself = notself;

    if (distsquared)
        params.distsquared = distsquared;
    else
        params.distsquared = mydistsq;

    params.user_callback = callback;
    params.user_callback_param = param;
    params.xtree = xtree;
    params.ytree = ytree;
    if (progress) {
        callbacks.start_results = rs_start_results;
        callbacks.start_extra = &params;
        params.user_progress = progress;
        params.user_progress_param = progress_param;
        params.ydone = 0;
    }

    dualtree_search(xtree, ytree, &callbacks);
}

static void rs_start_results(void* vparams,
                             kdtree_t* ytree, int ynode) {
    rs_params* p = (rs_params*)vparams;
    p->ydone += 1 + kdtree_right(ytree, ynode) - kdtree_left(ytree, ynode);
    if (p->user_progress)
        p->user_progress(p->user_progress_param, p->ydone);
}

static anbool rs_within_range(void* vparams,
                              kdtree_t* xtree, int xnode,
                              kdtree_t* ytree, int ynode) {
    rs_params* p = (rs_params*)vparams;
    //printf("rs_within_range: x node %i (parent %i) / y node %i (parent %i)\n", xnode, KD_PARENT(xnode), ynode, KD_PARENT(ynode));
    if (p->usemax &&
        kdtree_node_node_mindist2_exceeds(xtree, xnode, ytree, ynode,
                                          p->maxdistsq))
        return FALSE;

    if (p->usemin &&
        !kdtree_node_node_maxdist2_exceeds(xtree, xnode, ytree, ynode,
                                           p->mindistsq))
        return FALSE;

    return TRUE;
}

static void rs_handle_result(void* vparams,
                             kdtree_t* xtree, int xnode,
                             kdtree_t* ytree, int ynode) {
    // go through all pairs of points in this pair of nodes, checking
    // that each pair's distance lies within the required range.  Call the
    // user's callback function on each satisfying pair.
    int xl, xr, yl, yr;
    int x, y;
    rs_params* p = (rs_params*)vparams;
    int D = ytree->ndim;

    xl = kdtree_left (xtree, xnode);
    xr = kdtree_right(xtree, xnode);
    yl = kdtree_left (ytree, ynode);
    yr = kdtree_right(ytree, ynode);

    for (y=yl; y<=yr; y++) {
        double py[D];
        kdtree_copy_data_double(ytree, y, 1, py);
        // check if we can eliminate the whole box for this point...
        // HACK - can only do this if leaf nodes have bounding-boxes!
        if (!KD_IS_LEAF(xtree, xnode)) {
            if (p->usemax &&
                kdtree_node_point_mindist2_exceeds(xtree, xnode, py,
                                                   p->maxdistsq))
                continue;
            if (p->usemin &&
                !kdtree_node_point_maxdist2_exceeds(xtree, xnode, py,
                                                    p->mindistsq))
                continue;
        }
        for (x=xl; x<=xr; x++) {
            double d2;
            double px[D];
            if (p->notself && x == y)
                continue;
            kdtree_copy_data_double(xtree, x, 1, px);
            d2 = p->distsquared(px, py, D);
            //printf("eliminated point.\n");
            if ((p->usemax) && (d2 > p->maxdistsq))
                continue;
            if ((p->usemin) && (d2 < p->mindistsq))
                continue;
            p->user_callback(p->user_callback_param, x, y, d2);
        }
    }
}


/*
 void dualtree_rangecount(kdtree_t* x, kdtree_t* y,
 double mindist, double maxdist,
 dist2_function distsquared,
 int* counts) {
 printf("HACK - implement dualtree_rangecount.\n");
 }

 anbool rc_should_recurse(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode);
 void rc_handle_result(void* params, kdtree_node_t* search, kdtree_node_t* query);
 void rc_self_handle_result(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode);
 anbool rc_self_should_recurse(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode);
 void dualtree_rangecount(kdtree_t* xtree, kdtree_t* ytree,
 double mindist, double maxdist,
 int* counts) {
 dualtree_callbacks callbacks;
 rs_params params;

 memset(&callbacks, 0, sizeof(dualtree_callbacks));
 if (xtree == ytree) {
 callbacks.decision = rc_self_should_recurse;
 callbacks.result = rc_self_handle_result;
 } else {
 callbacks.decision = rc_should_recurse;
 callbacks.result = rc_handle_result;
 }
 callbacks.decision_extra = &params;
 callbacks.result_extra = &params;

 // set search params
 memset(&params, 0, sizeof(params));
 if (mindist == RANGESEARCH_NO_LIMIT) {
 params.usemin = 0;
 } else {
 params.usemin = 1;
 params.mindistsq = mindist * mindist;
 }
 if (maxdist == RANGESEARCH_NO_LIMIT) {
 params.usemax = 0;
 } else {
 params.usemax = 1;
 params.maxdistsq = maxdist * maxdist;
 }
 params.xtree = xtree;
 params.ytree = ytree;
 params.counts = counts;

 dualtree_search(xtree, ytree, &callbacks);
 }
 */

/*
 anbool rc_should_recurse(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode) {
 rs_params* p = (rs_params*)vparams;
 // does the bounding box partly overlap the desired range?
 if (p->usemax) {
 if (kdtree_node_node_mindist2_exceeds(p->xtree, xnode,
 p->ytree, ynode, p->maxdistsq))
 return FALSE;
 }
 if (p->usemin) {
 if (!kdtree_node_node_maxdist2_exceeds(p->xtree, xnode,
 p->ytree, ynode, p->mindistsq))
 return FALSE;
 }
 */
/*
 ;
 // HACK - it's not clear that it's advantageous to do this here...
 // (NOTE, if you decide to uncomment this, be sure to fix
 /    rc_self_should_recurse, since the action to take is different.)
 // is the bounding box fully within the desired range?
 if (p->usemin) {
 // compute min bound if it hasn't already been...
 if (!p->usemax)
 mindistsq = kdtree_node_node_mindist2(p->xtree, xnode, p->ytree, ynode);
 if (mindistsq < p->mindistsq)
 allinrange = FALSE;
 }
 if (allinrange && p->usemax) {
 if (!p->usemin)
 maxdistsq = kdtree_node_node_maxdist2(p->xtree, xnode, p->ytree, ynode);
 if (maxdistsq > p->maxdistsq)
 allinrange = FALSE;
 }
 if (allinrange) {
 // we can stop at this pair of nodes; no need to recurse any further.
 // for each Y point, increment its counter by the number of points in the X node.
 int NX, yl, yr, y;
 NX = kdtree_node_npoints(xnode);
 yl = ynode->l;
 yr = ynode->r;
 for (y=yl; y<=yr; y++) {
 int iy = p->ytree->perm[y];
 p->counts[iy] += NX;
 }
 return FALSE;
 }
 */
/*

 return TRUE;
 }

 void rc_handle_result(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode) {
 // go through all pairs of points in this pair of nodes, checking
 // that each pair's distance lies within the required range.
 int xl, xr, yl, yr;
 int x, y;
 rs_params* p = (rs_params*)vparams;
 int D = p->ytree->ndim;
 anbool allinrange = TRUE;
	
 // is the bounding box fully within the desired range?
 if (p->usemin) {
 if (!kdtree_node_node_mindist2_exceeds(p->xtree, xnode, p->ytree, ynode,
 p->mindistsq))
 allinrange = FALSE;
 }
 if (allinrange && p->usemax) {
 if (kdtree_node_node_maxdist2_exceeds(p->xtree, xnode, p->ytree, ynode, p->maxdistsq))
 allinrange = FALSE;
 }
 if (allinrange) {
 // for each Y point, increment its counter by the number of points in the X node.
 int NX, yl, yr, y;
 NX = kdtree_node_npoints(xnode);
 yl = ynode->l;
 yr = ynode->r;
 for (y=yl; y<=yr; y++) {
 p->counts[y] += NX;
 }
 return;
 }

 xl = xnode->l;
 xr = xnode->r;
 yl = ynode->l;
 yr = ynode->r;

 if (p->usemax && !p->usemin) {
 for (y=yl; y<=yr; y++) {
 double* py = p->ytree->data + y * D;
 for (x=xl; x<=xr; x++) {
 double* px;
 px = p->xtree->data + x * D;
 if (distsq_exceeds(px, py, D, p->maxdistsq))
 continue;
 p->counts[y]++;
 }
 }
 } else {
 for (y=yl; y<=yr; y++) {
 double* py = p->ytree->data + y * D;
 for (x=xl; x<=xr; x++) {
 double d2;
 double* px;
 px = p->xtree->data + x * D;
 d2 = distsq(px, py, D);
 if ((p->usemax) && (d2 > p->maxdistsq))
 continue;
 if ((p->usemin) && (d2 < p->mindistsq))
 continue;
 p->counts[y]++;
 }
 }
 }
 }





 anbool rc_self_should_recurse(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode) {
 if (xnode > ynode)
 return FALSE;
 return rc_should_recurse(vparams, xnode, ynode);
 }

 void rc_self_handle_result(void* vparams, kdtree_node_t* xnode, kdtree_node_t* ynode) {
 int xl, xr, yl, yr;
 int x, y;
 rs_params* p = (rs_params*)vparams;
 int D = p->ytree->ndim;

 if (xnode > ynode)
 return;

 if (xnode == ynode) {
 int x2;
 xl = xnode->l;
 xr = xnode->r;

 for (x=xl; x<=xr; x++) {
 double* px = p->xtree->data + x * D;
 for (x2=x+1; x2<=xr; x2++) {
 double d2;
 double* px2;
 px2 = p->xtree->data + x2 * D;
 d2 = distsq(px, px2, D);
 if ((p->usemax) && (d2 > p->maxdistsq))
 continue;
 if ((p->usemin) && (d2 < p->mindistsq))
 continue;
 p->counts[x]++;
 p->counts[x2]++;
 }
 // the diagonal...
 if ((p->usemin) && (0.0 < p->mindistsq))
 continue;
 p->counts[x]++;
 }
 return;
 }

 xl = xnode->l;
 xr = xnode->r;
 yl = ynode->l;
 yr = ynode->r;

 if (p->usemax && !p->usemin) {
 for (y=yl; y<=yr; y++) {
 double* py = p->ytree->data + y * D;
 for (x=xl; x<=xr; x++) {
 double* px;
 px = p->xtree->data + x * D;
 if (distsq_exceeds(px, py, D, p->maxdistsq))
 continue;
 p->counts[y]++;
 p->counts[x]++;
 }
 }
 } else {
 for (y=yl; y<=yr; y++) {
 double* py = p->ytree->data + y * D;
 for (x=xl; x<=xr; x++) {
 double d2;
 double* px;
 px = p->xtree->data + x * D;
 d2 = distsq(px, py, D);
 if ((p->usemax) && (d2 > p->maxdistsq))
 continue;
 if ((p->usemin) && (d2 < p->mindistsq))
 continue;
 p->counts[y]++;
 p->counts[x]++;
 }
 }
 }
 }

 */
