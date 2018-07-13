/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef KDTREE_NO_DUALTREE

#include "bl.h"
#include "errors.h"

typedef void (*rangesearch_callback)(void* user,
                                     kdtree_t* kd1, int ind1,
                                     kdtree_t* kd2, int ind2,
                                     double dist2);

static void dtrs_nodes(kdtree_t* xtree, kdtree_t* ytree,
                       int xnode, int ynode, double maxd2
                       rangesearch_callback cb, void* baton) {
}

/*
 At each step of the recursion, we have a query node ("ynode") and a
 list of candidate search nodes ("nodes" and "leaves" in the "xtree").
 
 General idea:
 - if we've hit a leaf in the "ytree", callback results; done.
 - if there are only leaves, no "x"/search nodes left, callback results; done.
 - for each element in x node list:
 -   if decision(xnode, ynode)
 -     add children of xnode to child search list
 - recurse on ynode's children with the child search list
 - empty the child search list

 The search order is depth-first, left-to-right in the "y" tree.

 */

static void dualtree_rs_recurse(kdtree_t* xtree, kdtree_t* ytree,
                                il* xnodes, il* xleaves,
                                bl* xnodebbs, bl* xleafbbs,
                                int ynode,
                                ttype* ybb,
                                double maxd2,
                                rangesearch_callback cb, void* baton) {
    int leafmarker;
    il* childnodes;
    int i, N;
    ttype oldbbval;
    ttype splitval;
    uint8_t splitdim;

    // if the query node is a leaf...
    if (KD_IS_LEAF(ytree, ynode)) {
        // ... then run the result function on each x node
        /*
         if (callbacks->start_results)
         callbacks->start_results(callbacks->start_extra, ytree, ynode);
         */
        if (cb) {
            // non-leaf nodes
            N = il_size(xnodes);
            for (i=0; i<N; i++)
                dtrs_nodes(xtree, ytree, il_get(xnodes, i), ynode, maxd2, cb, baton);
            // leaf nodes
            N = il_size(xleaves);
            for (i=0; i<N; i++)
                dtrs_nodes(xtree, ytree, il_get(xleaves, i), ynode, maxd2, cb, baton);
        }
        /*
         if (callbacks->end_results)
         callbacks->end_results(callbacks->end_extra, ytree, ynode);
         */
        return;
    }

    // if there are search leaves but no search nodes, run the result
    // function on each leaf.  (Note that the query node is not a leaf!)
    if (!il_size(xnodes)) {
        /*
         result_function result = callbacks->result;
         void* result_extra = callbacks->result_extra;
         if (callbacks->start_results)
         callbacks->start_results(callbacks->start_extra, ytree, ynode);
         */
        // leaf nodes
        if (result) {
            N = il_size(xleaves);
            for (i=0; i<N; i++)
                dtrs_nodes(xtree, ytree, il_get(xleaves, i), ynode, maxd2, cb, baton);
            //result(result_extra, xtree, il_get(leaves, i), ytree, ynode);
        }
        /*
         if (callbacks->end_results)
         callbacks->end_results(callbacks->end_extra, ytree, ynode);
         */
        return;
    }

    leafmarker = il_size(leaves);
    childnodes = il_new(256);

#define BBLO(bb, d) ((bb)[2*(d)])
#define BBHI(bb, d) ((bb)[(2*(d))+1])

    N = il_size(xnodes);
    for (i=0; i<N; i++) {
        int child1, child2;
        int xnode = il_get(xnodes, i);
        ttype* xbb = bl_access(xnodebbs, i);
        ttype* leftbb;
        ttype* rightbb;

        /*
         node-node range...
         if (!decision(decision_extra, xtree, xnode, ytree, ynode))
         continue;
         */
        split_dim_and_value(xtree, xnode, &splitdim, &splitval);
        child1 = KD_CHILD_LEFT(xnode);
        if (KD_IS_LEAF(xtree, child1)) {
            il_append(xleaves, child1);
            il_append(xleaves, child2);
            leftbb  = bl_append(xleafbbs, xbb);
            rightbb = bl_append(xleafbbs, xbb);
        } else {
            il_append(childnodes, child1);
            il_append(childnodes, child2);
            leftbb  = bl_append(xnodebbs, xbb);
            rightbb = bl_append(xnodebbs, xbb);
        }
        BBHI(leftbb,  splitdim) = splitval;
        BBLO(rightbb, splitdim) = splitval;
    }

    printf("dualtree: start left child of y node %i: %i\n", ynode, KD_CHILD_LEFT(ynode));
    // recurse on the Y children!
    split_dim_and_value(ytree, ynode, &splitdim, &splitval);
    // update y bb for the left child: max(splitdim) = splitval
    oldbbval = BBHI(ybb, splitdim);
    BBHI(ybb, splitdim) = splitval;
    dualtree_recurse(xtree, ytree, childnodes, leaves,
                     KD_CHILD_LEFT(ynode), callbacks);
    BBHI(ybb, splitdim) = oldbbval;
    printf("dualtree: done left child of y node %i: %i\n", ynode, KD_CHILD_LEFT(ynode));

    printf("dualtree: start right child of y node %i: %i\n", ynode, KD_CHILD_RIGHT(ynode));
    // update y bb for the right child: min(splitdim) = splitval
    oldbbval = BBLO(ybb, splitdim);
    BBLO(ybb, splitdim) = splitval;
    dualtree_recurse(xtree, ytree, childnodes, leaves,
                     KD_CHILD_RIGHT(ynode), callbacks);
    BBLO(ybb, splitdim) = oldbbval;
    printf("dualtree: done right child of y node %i: %i\n", ynode, KD_CHILD_LEFT(ynode));

    // put the "leaves" list back the way it was...
    il_remove_index_range(leaves, leafmarker, il_size(leaves)-leafmarker);
    il_free(childnodes);
}

#undef BBLO
#undef BBHI

int MANGLE(kdtree_dualtree_rangesearch)(kdtree_t* kd1, kdtree_t* kd2,
                                        double maxdist,
                                        rangesearch_callback callback, void* baton) {
    int xnode, ynode;
    il* nodes;
    il* leaves;

    if (kdtree_treetype(kd1) != kdtree_treetype(kd2)) {
        ERROR("Trees must be the same type.");
        return -1;
    }

    if (!kd1->split.any || !kd2->split.any) {
        ERROR("This function only supports splitting-plane trees.\n");
        return -1;
    }

    nodes = il_new(256);
    leaves = il_new(256);
    // root nodes.
    xnode = ynode = 0;
    if (KD_IS_LEAF(xtree, xnode))
        il_append(leaves, xnode);
    else
        il_append(nodes, xnode);

    dualtree_recurse(xtree, ytree, nodes, leaves, ynode, callbacks);

    il_free(nodes);
    il_free(leaves);
    return 0;
}





#endif

