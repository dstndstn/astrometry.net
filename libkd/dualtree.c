/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include "dualtree.h"
#include "bl.h"

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
static void dualtree_recurse(kdtree_t* xtree, kdtree_t* ytree,
                             il* nodes, il* leaves,
                             int ynode, dualtree_callbacks* callbacks) {

    // annoyances:
    //   -trees are of different heights, so we can reach the leaves of one
    //    before the leaves of the other.  if we hit a leaf in the query
    //    tree, just call the result function with all the search nodes,
    //    leaves or not.  if we hit a leaf in the search tree, add it to
    //    the leaf-list.
    //   -we want to share search lists between the children, but that means
    //    that the children can't modify the lists - or if they do, they
    //    have to undo any changes they make.  if we only append items, then
    //    we can undo changes by remembering the original list length and removing
    //    everything after it when we're done.
    int leafmarker;
    il* childnodes;
    decision_function decision;
    void* decision_extra;
    int i, N;

    // if the query node is a leaf...
    if (KD_IS_LEAF(ytree, ynode)) {
        // ... then run the result function on each search node.
        result_function result = callbacks->result;
        void* result_extra = callbacks->result_extra;

        if (callbacks->start_results)
            callbacks->start_results(callbacks->start_extra, ytree, ynode);

        if (result) {
            // non-leaf nodes
            N = il_size(nodes);
            for (i=0; i<N; i++)
                result(result_extra, xtree, il_get(nodes, i), ytree, ynode);
            // leaf nodes
            N = il_size(leaves);
            for (i=0; i<N; i++)
                result(result_extra, xtree, il_get(leaves, i), ytree, ynode);
        }
        if (callbacks->end_results)
            callbacks->end_results(callbacks->end_extra, ytree, ynode);

        return;
    }

    // if there are search leaves but no search nodes, run the result
    // function on each leaf.  (Note that the query node is not a leaf!)
    if (!il_size(nodes)) {
        result_function result = callbacks->result;
        void* result_extra = callbacks->result_extra;

        if (callbacks->start_results)
            callbacks->start_results(callbacks->start_extra, ytree, ynode);

        // leaf nodes
        if (result) {
            N = il_size(leaves);
            for (i=0; i<N; i++)
                result(result_extra, xtree, il_get(leaves, i), ytree, ynode);
        }

        if (callbacks->end_results)
            callbacks->end_results(callbacks->end_extra, ytree, ynode);

        return;
    }

    leafmarker = il_size(leaves);
    childnodes = il_new(32);
    decision = callbacks->decision;
    decision_extra = callbacks->decision_extra;


    N = il_size(nodes);
    for (i=0; i<N; i++) {
        int child1, child2;
        int xnode = il_get(nodes, i);
        if (!decision(decision_extra, xtree, xnode, ytree, ynode))
            continue;

        child1 = KD_CHILD_LEFT(xnode);
        child2 = KD_CHILD_RIGHT(xnode);

        if (KD_IS_LEAF(xtree, child1)) {
            il_append(leaves, child1);
            il_append(leaves, child2);
        } else {
            il_append(childnodes, child1);
            il_append(childnodes, child2);
        }
    }

    //printf("dualtree: start left child of y node %i is %i\n", ynode, KD_CHILD_LEFT(ynode));
    // recurse on the Y children!
    dualtree_recurse(xtree, ytree, childnodes, leaves,
                     KD_CHILD_LEFT(ynode), callbacks);
    //printf("dualtree: done left child of y node %i is %i\n", ynode, KD_CHILD_LEFT(ynode));
    //printf("dualtree: start right child of y node %i is %i\n", ynode, KD_CHILD_RIGHT(ynode));
    dualtree_recurse(xtree, ytree, childnodes, leaves,
                     KD_CHILD_RIGHT(ynode), callbacks);
    //printf("dualtree: done right child of y node %i is %i\n", ynode, KD_CHILD_LEFT(ynode));

    // put the "leaves" list back the way it was...
    il_remove_index_range(leaves, leafmarker, il_size(leaves)-leafmarker);
    il_free(childnodes);
}

void dualtree_search(kdtree_t* xtree, kdtree_t* ytree,
                     dualtree_callbacks* callbacks) {
    int xnode, ynode;
    il* nodes = il_new(32);
    il* leaves = il_new(32);
    // root nodes.
    xnode = ynode = 0;
    if (KD_IS_LEAF(xtree, xnode))
        il_append(leaves, xnode);
    else
        il_append(nodes, xnode);

    dualtree_recurse(xtree, ytree, nodes, leaves, ynode, callbacks);

    il_free(nodes);
    il_free(leaves);
}

