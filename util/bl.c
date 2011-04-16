/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2009 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

#include "bl.h"

#include "keywords.h"
// for qsort_r
#include "os-features.h"

#include "bl.ph"
#include "log.h"

static bl_node* bl_new_node(bl* list);
static void bl_free_node(bl_node* node);
static void bl_remove_from_node(bl* list, bl_node* node,
								bl_node* prev, int index_in_node);

// Defined in bl.ph (private header):
// free_node
// NODE_DATA
// NODE_CHARDATA
// NODE_INTDATA
// NODE_DOUBLEDATA

// Defined in bl.inc (inlined functions):
// bl_size
// bl_access
// il_size
// il_get

#define DEFINE_SORT 1
#define nl il
#define number int
#define NL_PRINT(x) printf("%i", x)
#include "bl-nl.c"
#undef nl
#undef number
#undef NL_PRINT

#define nl ll
#define number int64_t
#define NL_PRINT(x) printf("%lli", (long long int)x)
#include "bl-nl.c"
#undef nl
#undef number
#undef NL_PRINT

#define nl fl
#define number float
#define NL_PRINT(x) printf("%f", (float)x)
#include "bl-nl.c"
#undef nl
#undef number
#undef NL_PRINT

#define nl dl
#define number double
#define NL_PRINT(x) printf("%g", x)
#include "bl-nl.c"
#undef nl
#undef number
#undef NL_PRINT

#undef DEFINE_SORT
#define DEFINE_SORT 0
#define nl pl
#define number void*
#define NL_PRINT(x) printf("%p", x)
#include "bl-nl.c"
#undef nl
#undef number
#undef NL_PRINT
#undef DEFINE_SORT



Pure int bl_datasize(const bl* list) {
	if (!list)
		return 0;
	return list->datasize;
}

static void bl_sort_with_userdata(bl* list,
								  int (*compare)(const void* v1, const void* v2, void* userdata),
								  void* userdata);

struct funcandtoken {
	int (*compare)(const void* v1, const void* v2, void* userdata);
	void* userdata;
};
static int QSORT_COMPARISON_FUNCTION(qcompare, void* token, const void* v1, const void* v2) {
	struct funcandtoken* ft = token;
	return ft->compare(v1, v2, ft->userdata);
}

static void bl_sort_rec(bl* list, void* pivot,
						int (*compare)(const void* v1, const void* v2, void* userdata),
						void* userdata) {
	bl* less;
	bl* equal;
	bl* greater;
	int i;
    bl_node* node;

	// Empty case
	if (!list->head)
		return;

	// Base case: list with only one block.
	if (list->head == list->tail) {
		bl_node* node;
		struct funcandtoken ft;
		ft.compare = compare;
		ft.userdata = userdata;
		node = list->head;
		QSORT_R(NODE_DATA(node), node->N, list->datasize, &ft, qcompare);
		return;
	}

	less = bl_new(list->blocksize, list->datasize);
	equal = bl_new(list->blocksize, list->datasize);
	greater = bl_new(list->blocksize, list->datasize);
	for (node=list->head; node; node=node->next) {
		char* data = NODE_CHARDATA(node);
		for (i=0; i<node->N; i++) {
			int val = compare(data, pivot, userdata);
			if (val < 0)
				bl_append(less, data);
			else if (val > 0)
				bl_append(greater, data);
			else
				bl_append(equal, data);
			data += list->datasize;
		}
	}

    // recurse before freeing anything...
	bl_sort_with_userdata(less, compare, userdata);
	bl_sort_with_userdata(greater, compare, userdata);

	for (node=list->head; node;) {
        bl_node* next;
		next = node->next;
		bl_free_node(node);
		node = next;
    }
	list->head = NULL;
	list->tail = NULL;
	list->N = 0;
	list->last_access = NULL;
	list->last_access_n = 0;

	if (less->N) {
		list->head = less->head;
		list->tail = less->tail;
		list->N = less->N;
	}
	if (equal->N) {
		if (list->N) {
			list->tail->next = equal->head;
			list->tail = equal->tail;
		} else {
			list->head = equal->head;
			list->tail = equal->tail;
		}
		list->N += equal->N;
	}
	if (greater->N) {
		if (list->N) {
			list->tail->next = greater->head;
			list->tail = greater->tail;
		} else {
			list->head = greater->head;
			list->tail = greater->tail;
		}
		list->N += greater->N;
	}
	// note, these are supposed to be "free", not "bl_free"; we've stolen
	// the blocks, we're just freeing the headers.
	free(less);
	free(equal);
	free(greater);
}

static void bl_sort_with_userdata(bl* list,
								  int (*compare)(const void* v1, const void* v2, void* userdata),
								  void* userdata) {
	int ind;
	int N = list->N;
	if (N <= 1)
		return;
	// should do median-of-3/5/... to select pivot when N is large.
	ind = rand() % N;
	bl_sort_rec(list, bl_access(list, ind), compare, userdata);
}

static int sort_helper_bl(const void* v1, const void* v2, void* userdata) {
	int (*compare)(const void* v1, const void* v2) = userdata;
	return compare(v1, v2);
}

void bl_sort(bl* list, int (*compare)(const void* v1, const void* v2)) {
	bl_sort_with_userdata(list, sort_helper_bl, compare);
}

void bl_split(bl* src, bl* dest, int split) {
    bl_node* node;
    int nskipped;
    int ind;
    int ntaken = src->N - split;
    node = find_node(src, split, &nskipped);
    ind = split - nskipped;
    if (ind == 0) {
        // this whole node belongs to "dest".
        if (split) {
            // we need to get the previous node...
            bl_node* last = find_node(src, split-1, NULL);
            last->next = NULL;
            src->tail = last;
        } else {
            // we've removed everything from "src".
            src->head = NULL;
            src->tail = NULL;
        }
    } else {
        // create a new node to hold the second half of the items in "node".
        bl_node* newnode = bl_new_node(dest);
        newnode->N = (node->N - ind);
        newnode->next = node->next;
		memcpy(NODE_CHARDATA(newnode),
			   NODE_CHARDATA(node) + (ind * src->datasize),
			   newnode->N * src->datasize);
        node->N -= (node->N - ind);
        node->next = NULL;
        src->tail = node;
        // to make the code outside this block work...
        node = newnode;
    }

    // append it to "dest".
    if (dest->tail) {
        dest->tail->next = node;
        dest->N += ntaken;
    } else {
        dest->head = node;
        dest->tail = node;
        dest->N += ntaken;
    }

    // adjust "src".
    src->N -= ntaken;
    src->last_access = NULL;
}

void bl_init(bl* list, int blocksize, int datasize) {
	list->head = NULL;
	list->tail = NULL;
	list->N = 0;
	list->blocksize = blocksize;
	list->datasize  = datasize;
	list->last_access = NULL;
	list->last_access_n = 0;
}

bl* bl_new(int blocksize, int datasize) {
	bl* rtn;
	rtn = malloc(sizeof(bl));
	if (!rtn) {
		printf("Couldn't allocate memory for a bl.\n");
		return NULL;
	}
	bl_init(rtn, blocksize, datasize);
	return rtn;
}

void bl_free(bl* list) {
	if (!list) return;
	bl_remove_all(list);
	free(list);
}

void bl_remove_all(bl* list) {
	bl_node *n, *lastnode;
	lastnode = NULL;
	for (n=list->head; n; n=n->next) {
		if (lastnode)
			bl_free_node(lastnode);
		lastnode = n;
	}
	if (lastnode)
		bl_free_node(lastnode);
	list->head = NULL;
	list->tail = NULL;
	list->N = 0;
	list->last_access = NULL;
	list->last_access_n = 0;
}

void bl_remove_all_but_first(bl* list) {
	bl_node *n, *lastnode;
	lastnode = NULL;

	if (list->head) {
		for (n=list->head->next; n; n=n->next) {
			if (lastnode)
				bl_free_node(lastnode);
			lastnode = n;
		}
		if (lastnode)
			bl_free_node(lastnode);
		list->head->next = NULL;
		list->head->N = 0;
		list->tail = list->head;
	} else {
		list->head = NULL;
		list->tail = NULL;
	}
	list->N = 0;
	list->last_access = NULL;
	list->last_access_n = 0;
}

static void bl_remove_from_node(bl* list, bl_node* node,
								bl_node* prev, int index_in_node) {
	// if we're removing the last element at this node, then
	// remove this node from the linked list.
	if (node->N == 1) {
		// if we're removing the first node...
		if (prev == NULL) {
			list->head = node->next;
			// if it's the first and only node...
			if (list->head == NULL) {
				list->tail = NULL;
			}
		} else {
			// if we're removing the last element from
			// the tail node...
			if (node == list->tail) {
				list->tail = prev;
			}
			prev->next = node->next;
		}
		bl_free_node(node);
	} else {
		int ncopy;
		// just remove this element...
		ncopy = node->N - index_in_node - 1;
		if (ncopy > 0) {
			memmove(NODE_CHARDATA(node) + index_in_node * list->datasize,
					NODE_CHARDATA(node) + (index_in_node+1) * list->datasize,
					ncopy * list->datasize);
		}
		node->N--;
	}
	list->N--;
}

void bl_remove_index(bl* list, int index) {
	// find the node (and previous node) at which element 'index'
	// can be found.
	bl_node *node, *prev;
	int nskipped = 0;
	for (node=list->head, prev=NULL;
		 node;
		 prev=node, node=node->next) {

		if (index < (nskipped + node->N))
			break;
		nskipped += node->N;
	}
	assert(node);
	bl_remove_from_node(list, node, prev, index-nskipped);
	list->last_access = NULL;
	list->last_access_n = 0;
}

void bl_remove_index_range(bl* list, int start, int length) {
	// find the node (and previous node) at which element 'start'
	// can be found.
	bl_node *node, *prev;
	int nskipped = 0;
	list->last_access = NULL;
	list->last_access_n = 0;
	for (node=list->head, prev=NULL;
		 node;
		 prev=node, node=node->next) {

		if (start < (nskipped + node->N))
			break;

		nskipped += node->N;
	}

	// begin by removing any indices that are at the end of a block.
	if (start > nskipped) {
		// we're not removing everything at this node.
		int istart;
		int n;
		istart = start - nskipped;
		if ((istart + length) < node->N) {
			// we're removing a chunk of elements from the middle of this
			// block.  move elements from the end into the removed chunk.
			memmove(NODE_CHARDATA(node) + istart * list->datasize,
					NODE_CHARDATA(node) + (istart + length) * list->datasize,
					(node->N - (istart + length)) * list->datasize);
			// we're done!
			node->N -= length;
			list->N -= length;
			return;
		} else {
			// we're removing everything from 'istart' to the end of this
			// block.  just change the "N" values.
			n = (node->N - istart);
			node->N -= n;
			list->N -= n;
			length -= n;
			start += n;
			nskipped = start;
			prev = node;
			node = node->next;
		}
	}

	// remove complete blocks.
	for (;;) {
		int n;
		bl_node* todelete;
		if (length == 0 || length < node->N)
			break;
		// we're skipping this whole block.
		n = node->N;
		length -= n;
		start += n;
		list->N -= n;
		nskipped += n;
		todelete = node;
		node = node->next;
		bl_free_node(todelete);
	}
	if (prev)
		prev->next = node;
	else
		list->head = node;

	if (!node)
		list->tail = prev;

	// remove indices from the beginning of the last block.
	// note that we may have removed everything from the tail of the list,
	// no "node" may be null.
	if (node && length>0) {
		//printf("removing %i from end.\n", length);
		memmove(NODE_CHARDATA(node),
				NODE_CHARDATA(node) + length * list->datasize,
				(node->N - length) * list->datasize);
		node->N -= length;
		list->N -= length;
	}
}

static void clear_list(bl* list) {
	list->head = NULL;
	list->tail = NULL;
	list->N = 0;
	list->last_access = NULL;
	list->last_access_n = 0;
}

void bl_append_list(bl* list1, bl* list2) {
	list1->last_access = NULL;
	list1->last_access_n = 0;
	if (list1->datasize != list2->datasize) {
		printf("Error: cannot append bls with different data sizes!\n");
		assert(0);
		exit(0);
	}
	if (list1->blocksize != list2->blocksize) {
		printf("Error: cannot append bls with different block sizes!\n");
		assert(0);
		exit(0);
	}

	// if list1 is empty, then just copy over list2's head and tail.
	if (list1->head == NULL) {
		list1->head = list2->head;
		list1->tail = list2->tail;
		list1->N = list2->N;
		// remove everything from list2 (to avoid sharing nodes)
		clear_list(list2);
		return;
	}

	// if list2 is empty, then do nothing.
	if (list2->head == NULL)
		return;

	// otherwise, append list2's head to list1's tail.
	list1->tail->next = list2->head;
	list1->tail = list2->tail;
	list1->N += list2->N;
	// remove everything from list2 (to avoid sharing nodes)
	clear_list(list2);
}

static void bl_free_node(bl_node* node) {
	free(node);
}

static bl_node* bl_new_node(bl* list) {
	bl_node* rtn;
	// merge the mallocs for the node and its data into one malloc.
	rtn = malloc(sizeof(bl_node) + list->datasize * list->blocksize);
	if (!rtn) {
		printf("Couldn't allocate memory for a bl node!\n");
		return NULL;
	}
	//rtn->data = (char*)rtn + sizeof(bl_node);
	rtn->N = 0;
	rtn->next = NULL;
	return rtn;
}

static void bl_append_node(bl* list, bl_node* node) {
	node->next = NULL;
	if (!list->head) {
		// first node to be added.
		list->head = node;
		list->tail = node;
	} else {
		list->tail->next = node;
		list->tail = node;
	}
	list->N += node->N;
}

/*
 * Append an item to this bl node.  If this node is full, then create a new
 * node and insert it into the list.
 *
 * Returns the location where the new item was copied.
 */
void* bl_node_append(bl* list, bl_node* node, const void* data) {
	void* dest;
	if (node->N == list->blocksize) {
		// create a new node and insert it after the current node.
		bl_node* newnode;
		newnode = bl_new_node(list);
		newnode->next = node->next;
		node->next = newnode;
		if (list->tail == node)
			list->tail = newnode;
		node = newnode;
	}
	// space remains at this node.  add item.
	dest = NODE_CHARDATA(node) + node->N * list->datasize;
	if (data)
		memcpy(dest, data, list->datasize);
	node->N++;
	list->N++;
	return dest;
}

void* bl_append(bl* list, const void* data) {
	if (!list->tail)
		// empty list; create a new node.
		bl_append_node(list, bl_new_node(list));
	// append the item to the tail.  if the tail node is full, a new tail node may be created.
	return bl_node_append(list, list->tail, data);
}

void* bl_push(bl* list, const void* data) {
	return bl_append(list, data);
}

void bl_pop(bl* list, void* into) {
	bl_get(list, list->N-1, into);
    bl_remove_index(list, list->N-1);
}

void bl_print_structure(bl* list) {
	bl_node* n;
	printf("bl: head %p, tail %p, N %i\n", list->head, list->tail, list->N);
	for (n=list->head; n; n=n->next) {
		printf("[N=%i] ", n->N);
	}
	printf("\n");
}

void bl_get(bl* list, int n, void* dest) {
	char* src = bl_access(list, n);
	memcpy(dest, src, list->datasize);
}

static void bl_find_ind_and_element(bl* list, const void* data,
									int (*compare)(const void* v1, const void* v2),
									void** presult, int* pindex) {
	int lower, upper;
	int cmp = -2;
	void* result;
	lower = -1;
	upper = list->N;
	while (lower < (upper-1)) {
		int mid;
		mid = (upper + lower) / 2;
		cmp = compare(data, bl_access(list, mid));
		if (cmp >= 0) {
			lower = mid;
		} else {
			upper = mid;
		}
	}
	if (lower == -1 || compare(data, (result = bl_access(list, lower)))) {
		*presult = NULL;
		*pindex = -1;
		return;
	}
	*presult = result;
	*pindex = lower;
}

/**
 * Finds a node for which the given compare() function
 * returns zero when passed the given 'data' pointer
 * and elements from the list.
 */
void* bl_find(bl* list, const void* data,
			  int (*compare)(const void* v1, const void* v2)) {
	void* rtn;
	int ind;
	bl_find_ind_and_element(list, data, compare, &rtn, &ind);
	return rtn;
}

int bl_find_index(bl* list, const void* data,
				  int (*compare)(const void* v1, const void* v2)) {
	void* val;
	int ind;
	bl_find_ind_and_element(list, data, compare, &val, &ind);
	return ind;
}

int bl_insert_sorted(bl* list, const void* data,
					 int (*compare)(const void* v1, const void* v2)) {
	int lower, upper;
	lower = -1;
	upper = list->N;
	while (lower < (upper-1)) {
		int mid;
		int cmp;
		mid = (upper + lower) / 2;
		cmp = compare(data, bl_access(list, mid));
		if (cmp >= 0) {
			lower = mid;
		} else {
			upper = mid;
		}
	}
	bl_insert(list, lower+1, data);
	return lower+1;
}

int bl_insert_unique_sorted(bl* list, const void* data,
							int (*compare)(const void* v1, const void* v2)) {
	// This is just straightforward binary search - really should
	// use the block structure...
	int lower, upper;
	lower = -1;
	upper = list->N;
	while (lower < (upper-1)) {
		int mid;
		int cmp;
		mid = (upper + lower) / 2;
		cmp = compare(data, bl_access(list, mid));
		if (cmp >= 0) {
			lower = mid;
		} else {
			upper = mid;
		}
	}

	if (lower >= 0) {
		if (compare(data, bl_access(list, lower)) == 0) {
			return -1;
		}
	}
	bl_insert(list, lower+1, data);
	return lower+1;
}

void bl_set(bl* list, int index, const void* data) {
	bl_node* node;
	int nskipped;
	void* dataloc;

	node = find_node(list, index, &nskipped);
	dataloc = NODE_CHARDATA(node) + (index - nskipped) * list->datasize;
	memcpy(dataloc, data, list->datasize);
	// update the last_access member...
	list->last_access = node;
	list->last_access_n = nskipped;
}

/**
 * Insert the element "data" into the list, such that its index is "index".
 * All elements that previously had indices "index" and above are moved
 * one position to the right.
 */
void bl_insert(bl* list, int index, const void* data) {
	bl_node* node;
	int nskipped;

	if (list->N == index) {
		bl_append(list, data);
		return;
	}

	node = find_node(list, index, &nskipped);

	list->last_access = node;
	list->last_access_n = nskipped;

	// if the node is full:
	//   if we're inserting at the end of this node, then create a new node.
	//   else, shift all but the last element, add in this element, and 
	//     add the last element to a new node.
	if (node->N == list->blocksize) {
		int localindex, nshift;
		bl_node* next = node->next;
		bl_node* destnode;
		localindex = index - nskipped;

		// if the next node exists and is not full, then insert the overflowing
		// element at the front.  otherwise, create a new node.
		if (next && (next->N < list->blocksize)) {
			// shift the existing elements up by one position...
			memmove(NODE_CHARDATA(next) + list->datasize,
					NODE_CHARDATA(next),
					next->N * list->datasize);
			destnode = next;
		} else {
			// create and insert a new node.
			bl_node* newnode = bl_new_node(list);
			newnode->next = next;
			node->next = newnode;
			if (!newnode->next)
				list->tail = newnode;
			destnode = newnode;
		}

		if (localindex == node->N) {
			// the new element becomes the first element in the destination node.
			memcpy(NODE_CHARDATA(destnode), data, list->datasize);
		} else {
			// the last element in this node is added to the destination node.
			memcpy(NODE_CHARDATA(destnode), NODE_CHARDATA(node) + (node->N-1)*list->datasize, list->datasize);
			// shift the end portion of this node up by one...
			nshift = node->N - localindex - 1;
			memmove(NODE_CHARDATA(node) + (localindex+1) * list->datasize,
					NODE_CHARDATA(node) + localindex * list->datasize,
					nshift * list->datasize);
			// insert the new element...
			memcpy(NODE_CHARDATA(node) + localindex * list->datasize, data, list->datasize);
		}

		destnode->N++;
		list->N++;

	} else {
		// shift...
		int localindex, nshift;
		localindex = index - nskipped;
		nshift = node->N - localindex;
		memmove(NODE_CHARDATA(node) + (localindex+1) * list->datasize,
				NODE_CHARDATA(node) + localindex * list->datasize,
				nshift * list->datasize);
		// insert...
		memcpy(NODE_CHARDATA(node) + localindex * list->datasize,
			   data, list->datasize);
		node->N++;
		list->N++;
	}
}

void* bl_access_const(const bl* list, int n) {
	bl_node* node;
	int nskipped;
	node = find_node(list, n, &nskipped);
	// grab the element.
	return NODE_CHARDATA(node) + (n - nskipped) * list->datasize;
}

void bl_copy(bl* list, int start, int length, void* vdest) {
	bl_node* node;
	int nskipped;
	char* dest;
	if (length <= 0)
		return;
	node = find_node(list, start, &nskipped);

	// we've found the node containing "start".  keep copying elements and
	// moving down the list until we've copied all "length" elements.
	dest = vdest;
	while (length > 0) {
		int take, avail;
		char* src;
		// number of elements we want to take.
		take = length;
		// number of elements available at this node.
		avail = node->N - (start - nskipped);
		if (take > avail)
			take = avail;
		src = NODE_CHARDATA(node) + (start - nskipped) * list->datasize;
		memcpy(dest, src, take * list->datasize);

		dest += take * list->datasize;
		start += take;
		length -= take;
		nskipped += node->N;
		node = node->next;
	}
	// update the last_access member...
	list->last_access = node;
	list->last_access_n = nskipped;
}

int bl_check_consistency(bl* list) {
	bl_node* node;
	int N;
	int tailok = 1;
	int nempty = 0;
	int nnull = 0;
	
	// if one of head or tail is NULL, they had both better be NULL!
	if (!list->head)
		nnull++;
	if (!list->tail)
		nnull++;
	if (nnull == 1) {
		fprintf(stderr, "bl_check_consistency: head is %p, and tail is %p.\n",
				list->head, list->tail);
		return 1;
	}

	N = 0;
	for (node=list->head; node; node=node->next) {
		N += node->N;
		if (!node->N) {
			// this block is empty.
			nempty++;
		}
		// are we at the last node?
		if (!node->next) {
			tailok = (list->tail == node) ? 1 : 0;
		}
	}
	if (!tailok) {
		fprintf(stderr, "bl_check_consistency: tail pointer is wrong.\n");
		return 1;
	}
	if (nempty) {
		fprintf(stderr, "bl_check_consistency: %i empty blocks.\n", nempty);
		return 1;
	}
	if (N != list->N) {
		fprintf(stderr, "bl_check_consistency: list->N is %i, but sum of blocks is %i.\n",
				list->N, N);
		return 1;
	}
	return 0;
}

int bl_check_sorted(bl* list,
					int (*compare)(const void* v1, const void* v2),
					int isunique) {
	int i, N;
	int nbad = 0;
	void* v2 = NULL;
	N = bl_size(list);
	if (N)
		v2 = bl_access(list, 0);
	for (i=1; i<N; i++) {
		void* v1;
		int cmp;
		v1 = v2;
		v2 = bl_access(list, i);
		cmp = compare(v1, v2);
		if (isunique) {
			if (cmp >= 0) {
				nbad++;
			}
		} else {
			if (cmp > 0) {
				nbad++;
			}
		}
	}
	if (nbad) {
		fprintf(stderr, "bl_check_sorted: %i are out of order.\n", nbad);
		return 1;
	}
	return 0;
}

static void memswap(void* v1, void* v2, int len) {
	unsigned char tmp;
	unsigned char* c1 = v1;
	unsigned char* c2 = v2;
	int i;
	for (i=0; i<len; i++) {
		tmp = *c1;
		*c1 = *c2;
		*c2 = tmp;
		c1++;
		c2++;
	}
}

void bl_reverse(bl* list) {
	// reverse each block, and reverse the order of the blocks.
	pl* blocks;
	bl_node* node;
	bl_node* lastnode;
	int i;

	// reverse each block
	blocks = pl_new(256);
	for (node=list->head; node; node=node->next) {
		for (i=0; i<(node->N/2); i++) {
			memswap(NODE_CHARDATA(node) + i * list->datasize,
					NODE_CHARDATA(node) + (node->N - 1 - i) * list->datasize,
					list->datasize);
		}
		pl_append(blocks, node);
	}

	// reverse the blocks
	lastnode = NULL;
	for (i=pl_size(blocks)-1; i>=0; i--) {
		node = pl_get(blocks, i);
		if (lastnode)
			lastnode->next = node;
		lastnode = node;
	}
	if (lastnode)
		lastnode->next = NULL;
	pl_free(blocks);

	// swap head and tail
	node = list->head;
	list->head = list->tail;
	list->tail = node;

	list->last_access = NULL;
	list->last_access_n = 0;
}

void* bl_extend(bl* list) {
	return bl_append(list, NULL);
}



// special-case pointer list accessors...
int bl_compare_pointers_ascending(const void* v1, const void* v2) {
    void* p1 = *(void**)v1;
    void* p2 = *(void**)v2;
    if (p1 > p2) return 1;
    else if (p1 < p2) return -1;
    else return 0;
}

void  pl_free_elements(pl* list) {
	int i;
	for (i=0; i<pl_size(list); i++) {
		free(pl_get(list, i));
	}
}

// dereference one level...
static int sort_helper_pl(const void* v1, const void* v2, void* userdata) {
	const void* p1 = *((const void**)v1);
	const void* p2 = *((const void**)v2);
	int (*compare)(const void* p1, const void* p2) = userdata;
	return compare(p1, p2);
}

void  pl_sort(pl* list, int (*compare)(const void* v1, const void* v2)) {
	bl_sort_with_userdata(list, sort_helper_pl, compare);
}

int pl_insert_sorted(pl* list, const void* data, int (*compare)(const void* v1, const void* v2)) {
	// we don't just call bl_insert_sorted because then we end up passing
	// "void**" rather than "void*" args to the compare function, which 
	// violates the principle of least surprise.
	int lower, upper;
	lower = -1;
	upper = list->N;
	while (lower < (upper-1)) {
		int mid;
		int cmp;
		mid = (upper + lower) / 2;
		cmp = compare(data, pl_get(list, mid));
		if (cmp >= 0) {
			lower = mid;
		} else {
			upper = mid;
		}
	}
	bl_insert(list, lower+1, &data);
	return lower+1;
}

/*
 void pl_set(pl* list, int index, void* data) {
 int i;
 int nadd = (index+1) - list->N;
 if (nadd > 0) {
 // enlarge the list to hold 'nadd' more entries.
 for (i=0; i<nadd; i++) {
 pl_append(list, NULL);
 }
 }
 bl_set(list, index, &data);
 }
 */

void sl_remove_duplicates(sl* lst) {
	int i, j;
	for (i=0; i<sl_size(lst); i++) {
		const char* s1 = sl_get(lst, i);
		for (j=i+1; j<sl_size(lst); j++) {
			const char* s2 = sl_get(lst, j);
			if (strcmp(s1, s2) == 0) {
				sl_remove(lst, j);
				j--;
			}
		}
	}
}

sl* sl_new(int blocksize) {
	pl* lst = pl_new(blocksize);
	assert(lst);
	return lst;
}

void sl_init2(sl* list, int blocksize) {
	pl_init(list, blocksize);
}

void sl_free2(sl* list) {
	int i;
	if (!list) return;
	for (i=0; i<sl_size(list); i++)
		free(sl_get(list, i));
	bl_free(list);
}

sl* sl_split(sl* lst, const char* str, const char* sepstring) {
    int seplen;
    const char* s;
    char* next_sep;
    if (!lst)
        lst = sl_new(4);
    seplen = strlen(sepstring);
    s = str;
    while (s && *s) {
        next_sep = strstr(s, sepstring);
        if (!next_sep) {
            sl_append(lst, s);
            break;
        }
        //logverb("Appending: '%.*s'\n", (int)(next_sep - s), s);
        sl_appendf(lst, "%.*s", (int)(next_sep - s), s);
        s = next_sep + seplen;
    }
    return lst;
}

void sl_free_nonrecursive(sl* list) {
	bl_free(list);
}

void sl_append_contents(sl* dest, sl* src) {
	int i;
	if (!src)
		return;
	for (i=0; i<sl_size(src); i++) {
		char* str = sl_get(src, i);
		sl_append(dest, str);
	}
}

int sl_index_of(sl* lst, const char* str) {
    int i;
    for (i=0; i<sl_size(lst); i++) {
        char* s = sl_get(lst, i);
        if (strcmp(s, str) == 0)
            return i;
    }
    return -1;
}

int sl_last_index_of(sl* lst, const char* str) {
    int i;
    for (i=sl_size(lst)-1; i>=0; i--) {
        char* s = sl_get(lst, i);
        if (strcmp(s, str) == 0)
            return i;
    }
    return -1;
}

// Returns 0 if the string is not in the sl, 1 otherwise.
// (same as sl_index_of(lst, str) > -1)
int sl_contains(sl* lst, const char* str) {
    return (sl_index_of(lst, str) > -1);
}

void sl_reverse(sl* list) {
	bl_reverse(list);
}

char* sl_append(sl* list, const char* data) {
	char* copy;
	if (data) {
		copy = strdup(data);
		assert(copy);
	} else
		copy = NULL;
	pl_append(list, copy);
	return copy;
}

void sl_append_array(sl* list, const char**strings, int n) {
	int i;
	for (i=0; i<n; i++)
		sl_append(list, strings[i]);
}

void sl_append_nocopy(sl* list, const char* data) {
	pl_append(list, data);
}

char* sl_push(sl* list, const char* data) {
	char* copy = strdup(data);
	pl_push(list, copy);
	return copy;
}

char* sl_pop(sl* list) {
	return pl_pop(list);
}

char* sl_get(sl* list, int n) {
	return pl_get(list, n);
}

char* sl_get_const(const sl* list, int n) {
	return pl_get_const(list, n);
}

char* sl_set(sl* list, int index, const char* value) {
	char* copy;
	assert(index >= 0);
	copy = strdup(value);
	if (index < list->N) {
		// we're replacing an existing value - free it!
		free(sl_get(list, index));
		bl_set(list, index, &copy);
	} else {
		// pad
		int i;
		for (i=list->N; i<index; i++)
			sl_append_nocopy(list, NULL);
		sl_append(list, copy);
	}
	return copy;
}

int sl_check_consistency(sl* list) {
	return bl_check_consistency(list);
}

char* sl_insert(sl* list, int indx, const char* data) {
	char* copy = strdup(data);
	bl_insert(list, indx, &copy);
	return copy;
}

void sl_insert_nocopy(sl* list, int indx, const char* str) {
	bl_insert(list, indx, &str);
}

void sl_remove_from(sl* list, int start) {
    sl_remove_index_range(list, start, sl_size(list) - start);
}

int sl_remove_string(sl* list, const char* string) {
    return pl_remove_value(list, string);
}

char* sl_remove_string_bycaseval(sl* list, const char* string) {
	int N = sl_size(list);
	int i;
	for (i=0; i<N; i++) {
        char* str = sl_get(list, i);
		if (strcasecmp(str, string) == 0) {
            char* s = sl_get(list, i);
			sl_remove(list, i);
            return s;
		}
	}
	return NULL;
}

int sl_remove_string_byval(sl* list, const char* string) {
	int N = sl_size(list);
	int i;
	for (i=0; i<N; i++) {
        char* str = sl_get(list, i);
		if (strcmp(str, string) == 0) {
			sl_remove(list, i);
			return i;
		}
	}
	return -1;
}

void sl_remove_index_range(sl* list, int start, int length) {
    int i;
    assert(list);
    assert(start + length <= list->N);
    assert(start >= 0);
    assert(length >= 0);
    for (i=0; i<length; i++) {
        char* str = sl_get(list, start + i);
        free(str);
    }
    bl_remove_index_range(list, start, length);
}

void sl_remove(sl* list, int index) {
    bl_remove_index(list, index);
}

void  sl_remove_all(sl* list) {
	int i;
	if (!list) return;
	for (i=0; i<sl_size(list); i++)
		free(pl_get(list, i));
	bl_remove_all(list);
}

void   sl_merge_lists(sl* list1, sl* list2) {
	bl_append_list(list1, list2);
}

void sl_print(sl* list) {
	bl_node* n;
	int i;
	for (n=list->head; n; n=n->next) {
		printf("[\n");
		for (i=0; i<n->N; i++)
			printf("  \"%s\"\n", ((char**)NODE_DATA(n))[i]);
		printf("]\n");
	}
}

static char* sljoin(sl* list, const char* join, int forward) {
    int start, end, inc;

	int len = 0;
	int i, N;
	char* rtn;
	int offset;
	int JL;

    if (sl_size(list) == 0)
        return strdup("");

    // step through the list forward or backward?
    if (forward) {
        start = 0;
        end = sl_size(list);
        inc = 1;
    } else {
        start = sl_size(list) - 1;
        end = -1;
        inc = -1;
    }

	JL = strlen(join);
	N = sl_size(list);
	for (i=0; i<N; i++)
		len += strlen(sl_get(list, i));
	len += ((N-1) * JL);
	rtn = malloc(len + 1);
	if (!rtn)
		return rtn;
	offset = 0;
	for (i=start; i!=end; i+= inc) {
		char* str = sl_get(list, i);
		int L = strlen(str);
		if (i != start) {
			memcpy(rtn + offset, join, JL);
			offset += JL;
		}
		memcpy(rtn + offset, str, L);
		offset += L;
	}
	assert(offset == len);
	rtn[offset] = '\0';
	return rtn;
}

char*  sl_join(sl* list, const char* join) {
    return sljoin(list, join, 1);
}

char*  sl_join_reverse(sl* list, const char* join) {
    return sljoin(list, join, 0);
}

char*  sl_implode(sl* list, const char* join) {
    return sl_join(list, join);
}

char* sl_appendf(sl* list, const char* format, ...) {
	char* str;
    va_list lst;
    va_start(lst, format);
    str = sl_appendvf(list, format, lst);
    va_end(lst);
	return str;
}

char* sl_appendvf(sl* list, const char* format, va_list va) {
	char* str;
    if (vasprintf(&str, format, va) == -1)
		return NULL;
	sl_append_nocopy(list, str);
    return str;
}

char* sl_insert_sortedf(sl* list, const char* format, ...) {
    va_list lst;
	char* str;
    va_start(lst, format);
    if (vasprintf(&str, format, lst) == -1)
		return NULL;
	sl_insert_sorted_nocopy(list, str);
    va_end(lst);
	return str;
}

char* sl_insertf(sl* list, int index, const char* format, ...) {
    va_list lst;
	char* str;
    va_start(lst, format);
    if (vasprintf(&str, format, lst) == -1)
		return NULL;
	sl_insert_nocopy(list, index, str);
    va_end(lst);
	return str;
}

int bl_compare_strings_ascending(const void* v1, const void* v2) {
    const char* str1 = v1;
    const char* str2 = v2;
    return strcoll(str1, str2);
}

void sl_insert_sorted_nocopy(sl* list, const char* string) {
    pl_insert_sorted(list, string, bl_compare_strings_ascending);
}

char* sl_insert_sorted(sl* list, const char* string) {
    char* copy = strdup(string);
    pl_insert_sorted(list, copy, bl_compare_strings_ascending);
    return copy;
}

#define InlineDefine InlineDefineC
#include "bl.inc"
#undef InlineDefine

