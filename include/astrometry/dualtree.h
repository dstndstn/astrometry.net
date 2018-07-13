/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 \file

 Dual-tree search.

 Input:
 -a search tree
 -a query tree
 -a function that takes two nodes and returns true if the result set should
 contain that pair of nodes.
 -an extra value that will be passed to the decision function.
 -a function that is called for each pair of leaf nodes.
 (** actually, at least one of the nodes will be a leaf. **)
 -an extra value that will be passed to the leaf-node function

 The query tree is a kd-tree built out of the points you want to query.

 The search and query trees can be the same tree.
 */

#include "astrometry/starutil.h"
#include "astrometry/kdtree.h"

typedef anbool (*decision_function)(void* extra, kdtree_t* searchtree, int searchnode,
                                    kdtree_t* querytree, int querynode);
typedef void (*start_of_results_function)(void* extra, kdtree_t* querytree, int querynode);
typedef void (*result_function)(void* extra, kdtree_t* searchtree, int searchnode,
                                kdtree_t* querytree, int querynode);
typedef void (*end_of_results_function)(void* extra, kdtree_t* querytree, int querynode);

struct dualtree_callbacks {
    decision_function          decision;
    void*                      decision_extra;
    start_of_results_function  start_results;
    void*                      start_extra;
    result_function            result;
    void*                      result_extra;
    end_of_results_function    end_results;
    void*                      end_extra;
};
typedef struct dualtree_callbacks dualtree_callbacks;

void dualtree_search(kdtree_t* search, kdtree_t* query,
                     dualtree_callbacks* callbacks);


