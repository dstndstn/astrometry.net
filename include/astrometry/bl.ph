/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

/// Private header file shared between bl.inc and bl.c

InlineDeclare
bl_node* find_node(const bl* list, size_t n, size_t* rtn_nskipped);

// data follows the bl_node*.
#define NODE_DATA(node) ((void*)(((bl_node*)(node)) + 1))
#define NODE_CHARDATA(node) ((char*)(((bl_node*)(node)) + 1))
#define NODE_INTDATA(node) ((int*)(((bl_node*)(node)) + 1))
#define NODE_DOUBLEDATA(node) ((double*)(((bl_node*)(node)) + 1))

#define bl_free_node free
