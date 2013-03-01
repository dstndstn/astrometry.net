/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef BT_H
#define BT_H

#include "keywords.h"
#include "an-bool.h"
/*
  We distinguish between "branch" (ie, internal) nodes and "leaf" nodes
  because leaf nodes can be much smaller.  Since there are a lot of leaves,
  the space savings can be considerable.

  The data owned by a leaf node follows right after the leaf struct
  itself.
 */

struct bt_leaf {
	// always 1; must be the first element in the struct.
	unsigned char isleaf;
	// number of data elements.
	short N;
	// data follows implicitly.
};
typedef struct bt_leaf bt_leaf;

struct bt_branch {
	// always 0; must be the first element in the struct.
	unsigned char isleaf;
	// AVL balance
	signed char balance;

	//struct bt_node* children[2];
	union bt_node* children[2];

	// the leftmost leaf node in this subtree.
	bt_leaf* firstleaf;

	// number of element in this subtree.
	int N;
};
typedef struct bt_branch bt_branch;

union bt_node {
	bt_leaf leaf;
	bt_branch branch;
};
typedef union bt_node bt_node;

struct bt {
	bt_node* root;
	int datasize;
	int blocksize;
	int N;
};
typedef struct bt bt;

typedef int (*compare_func)(const void* v1, const void* v2);

typedef int (*compare_func_2)(const void* v1, const void* v2, void* token);

Malloc bt* bt_new(int datasize, int blocksize);

void bt_free(bt* tree);

Pure //Inline
int bt_size(bt* tree);

anbool bt_insert(bt* tree, void* data, anbool unique, compare_func compare);

anbool bt_insert2(bt* tree, void* data, anbool unique, compare_func_2 compare, void* token);

anbool bt_contains(bt* tree, void* data, compare_func compare);

anbool bt_contains2(bt* tree, void* data, compare_func_2 compare, void* token);

void* bt_access(bt* tree, int index);

void bt_print(bt* tree, void (*print_element)(void* val));

void bt_print_structure(bt* tree, void (*print_element)(void* val));

int bt_height(bt* tree);

int bt_count_leaves(bt* tree);

int bt_check(bt* tree);

#endif
