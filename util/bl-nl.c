/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

/**
 Defined:

 --nl
 --number
 --NL_PRINT(x)  prints number 'x'

 Note:
 --You can't declare multiple "number" variables like this:
     number n1, n2;
   Instead, do:
     number n1;
     number n2;
   This is because "number" may be a pointer type.
 */

#include "bl-nl.ph"

#define NODE_NUMDATA(node) ((number*)NODE_DATA(node))

number* NLF(to_array)(nl* list) {
	number* arr;
	int N;
	if (!list)
		return NULL;
	N = NLF(size)(list);
	arr = malloc(N * sizeof(number));
	bl_copy(list, 0, N, arr);
	return arr;
}

static int NLF(compare_ascending)(const void* v1, const void* v2) {
    number i1 = *(number*)v1;
    number i2 = *(number*)v2;
    if (i1 > i2) return 1;
    else if (i1 < i2) return -1;
    else return 0;
}

static int NLF(compare_descending)(const void* v1, const void* v2) {
    number i1 = *(number*)v1;
    number i2 = *(number*)v2;
    if (i1 > i2) return -1;
    else if (i1 < i2) return 1;
    else return 0;
}

#define InlineDefine InlineDefineC
#include "bl-nl.inc"
#undef InlineDefine

void NLF(reverse)(nl* list) {
	bl_reverse(list);
}

void NLF(append_array)(nl* list, const number* data, int ndata) {
	int i;
	for (i=0; i<ndata; i++)
		NLF(append)(list, data[i]);
}

nl* NLF(merge_ascending)(nl* list1, nl* list2) {
	nl* res;
	int i1, i2, N1, N2;
	number v1 = 0;
	number v2 = 0;
	unsigned char getv1, getv2;
	if (!list1)
		return NLF(dupe)(list2);
	if (!list2)
		return NLF(dupe)(list1);
	if (!NLF(size)(list1))
		return NLF(dupe)(list2);
	if (!NLF(size)(list2))
		return NLF(dupe)(list1);

	res = NLF(new)(list1->blocksize);
	N1 = NLF(size)(list1);
	N2 = NLF(size)(list2);
	i1 = i2 = 0;
	getv1 = getv2 = 1;
	while (i1 < N1 && i2 < N2) {
		if (getv1) {
			v1 = NLF(get)(list1, i1);
			getv1 = 0;
		}
		if (getv2) {
			getv2 = 0;
			v2 = NLF(get)(list2, i2);
		}
		if (v1 <= v2) {
			NLF(append)(res, v1);
			i1++;
			getv1 = 1;
		} else {
			NLF(append)(res, v2);
			i2++;
			getv2 = 1;
		}
	}
	for (; i1<N1; i1++)
		NLF(append)(res, NLF(get)(list1, i1));
	for (; i2<N2; i2++)
		NLF(append)(res, NLF(get)(list2, i2));
	return res;
}

#if DEFINE_SORT
void NLF(sort)(nl* list, int ascending) {
	bl_sort(list, ascending ? NLF(compare_ascending) : NLF(compare_descending));
}
#endif

void NLF(remove_all_reuse)(nl* list) {
	bl_remove_all_but_first(list);
}

int  NLF(find_index_ascending)(nl* list, const number value) {
	return bl_find_index(list, &value, NLF(compare_ascending));
}

int NLF(check_consistency)(nl* list) {
	return bl_check_consistency(list);
}

int NLF(check_sorted_ascending)(nl* list,
							  int isunique) {
	return bl_check_sorted(list, NLF(compare_ascending), isunique);
}

int NLF(check_sorted_descending)(nl* list,
							   int isunique) {
	return bl_check_sorted(list, NLF(compare_descending), isunique);
}

void NLF(remove)(nl* nlist, int index) {
    bl_remove_index(nlist, index);
}

number NLF(pop)(nl* nlist) {
    number ret = NLF(get)(nlist, nlist->N-1);
    bl_remove_index(nlist, nlist->N-1);
    return ret;
}

nl* NLF(dupe)(nl* nlist) {
    nl* ret = NLF(new)(nlist->blocksize);
    int i;
    for (i=0; i<nlist->N; i++)
        NLF(push)(ret, NLF(get)(nlist, i));
    return ret;
}

int NLF(remove_value)(nl* nlist, const number value) {
    bl* list = nlist;
	bl_node *node, *prev;
	int istart = 0;
	for (node=list->head, prev=NULL;
		 node;
		 prev=node, node=node->next) {
		int i;
		number* idat;
		idat = NODE_DATA(node);
		for (i=0; i<node->N; i++)
			if (idat[i] == value) {
				bl_remove_from_node(list, node, prev, i);
				list->last_access = prev;
				list->last_access_n = istart;
				return istart + i;
			}
		istart += node->N;
	}
	return -1;
}

void NLF(remove_all)(nl* list) {
	bl_remove_all(list);
}

void NLF(remove_index_range)(nl* list, int start, int length) {
	bl_remove_index_range(list, start, length);
}

void NLF(set)(nl* list, int index, const number value) {
	bl_set(list, index, &value);
}

/*
 void dl_set(dl* list, int index, double value) {
 int i;
 int nadd = (index+1) - list->N;
 if (nadd > 0) {
 // enlarge the list to hold 'nadd' more entries.
 for (i=0; i<nadd; i++) {
 dl_append(list, 0.0);
 }
 }
 bl_set(list, index, &value);
 }
 */

nl* NLF(new)(int blocksize) {
	return bl_new(blocksize, sizeof(number));
}

void NLF(new_existing)(nl* list, int blocksize) {
	bl_init(list, blocksize, sizeof(number));
}

void NLF(init)(nl* list, int blocksize) {
	bl_init(list, blocksize, sizeof(number));
}

void NLF(free)(nl* list) {
	bl_free(list);
}

void NLF(push)(nl* list, const number data) {
	bl_append(list, &data);
}

number* NLF(append)(nl* list, const number data) {
	return bl_append(list, &data);
}

void NLF(append_list)(nl* list, nl* list2) {
    int i, N;
    N = NLF(size)(list2);
    for (i=0; i<N; i++)
        NLF(append)(list, NLF(get)(list2, i));
}

void NLF(merge_lists)(nl* list1, nl* list2) {
	bl_append_list(list1, list2);
}

static int NLF(binarysearch)(bl_node* node, const number n) {
	number* iarray = NODE_NUMDATA(node);
	int lower = -1;
	int upper = node->N;
	int mid;
	while (lower < (upper-1)) {
		mid = (upper + lower) / 2;
		if (n >= iarray[mid])
			lower = mid;
		else
			upper = mid;
	}
	return lower;
}

// find the first node for which n <= the last element.
static bl_node* NLF(findnodecontainingsorted)(const nl* list, const number n,
											  int* p_nskipped) {
	bl_node *node;
	int nskipped;
	//bl_node *prev;
	//int prevnskipped;

	// check if we can use the jump accessor or if we have to start at
	// the beginning...
	if (list->last_access && list->last_access->N &&
		// is the value we're looking for >= the first element?
		(n >= *NODE_NUMDATA(list->last_access))) {
		node = list->last_access;
		nskipped = list->last_access_n;
	} else {
		node = list->head;
		nskipped = 0;
	}

	/*
	 // find the first node for which n < the first element.  The
	 // previous node will contain the value (if it exists).
	 for (prev=node, prevnskipped=nskipped;
	 node && (n < *NODE_NUMDATA(node));) {
	 prev=node;
	 prevnskipped=nskipped;
	 nskipped+=node->N;
	 node=node->next;
	 }
	 if (prev && n <= NODE_NUMDATA(prev)[prev->N-1]) {
	 if (p_nskipped)
	 *p_nskipped = prevnskipped;
	 return prev;
	 }
	 if (node && n <= NODE_NUMDATA(node)[node->N-1]) {
	 if (p_nskipped)
	 *p_nskipped = nskipped;
	 return node;
	 }
	 return NULL;
	 */
	/*
	 if (!node && prev && n > NODE_NUMDATA(prev)[prev->N-1])
	 return NULL;
	 if (p_nskipped)
	 *p_nskipped = prevnskipped;
	 return prev;
	 */

	for (; node && (n > NODE_NUMDATA(node)[node->N-1]); node=node->next)
		nskipped += node->N;
	if (p_nskipped)
		*p_nskipped = nskipped;
	return node;
}

static int NLF(insertascending)(nl* list, const number n, int unique) {
	bl_node *node;
	int ind;
	int nskipped;

	node = NLF(findnodecontainingsorted)(list, n, &nskipped);
	if (!node) {
		NLF(append)(list, n);
		return list->N-1;
	}

	/*
	for (; node && (n > NODE_NUMDATA(node)[node->N-1]); node=node->next)
		nskipped += node->N;
	if (!node) {
		// either we're adding the first element, or we're appending since
		// n is bigger than the largest element in the list.
		NLF(append)(list, n);
		return list->N-1;
	}
	 */

	// find where in the node it should be inserted...
	ind = 1 + NLF(binarysearch)(node, n);

    // check if it's a duplicate...
	if (unique && ind > 0 && (n == NODE_NUMDATA(node)[ind-1]))
		return -1;

	// set the jump accessors...
	list->last_access = node;
	list->last_access_n = nskipped;
	// ... so that this runs in O(1).
	bl_insert(list, nskipped + ind, &n);
	return nskipped + ind;
}

int NLF(insert_ascending)(nl* list, const number n) {
	return NLF(insertascending)(list, n, 0);
}

int NLF(insert_unique_ascending)(nl* list, const number n) {
	return NLF(insertascending)(list, n, 1);
}

int NLF(insert_descending)(nl* list, const number n) {
    return bl_insert_sorted(list, &n, NLF(compare_descending));
}

void NLF(insert)(nl* list, int indx, const number data) {
	bl_insert(list, indx, &data);
}

void NLF(copy)(nl* list, int start, int length, number* vdest) {
	bl_copy(list, start, length, vdest);
}

void NLF(print)(nl* list) {
	bl_node* n;
	int i;
	for (n=list->head; n; n=n->next) {
		printf("[ ");
		for (i=0; i<n->N; i++) {
			if (i > 0)
				printf(", ");
			NL_PRINT(NODE_NUMDATA(n)[i]);
		}
		printf("] ");
	}
}

int  NLF(index_of)(nl* list, const number data) {
	bl_node* n;
	int i;
	number* idata;
	int npast = 0;
	for (n=list->head; n; n=n->next) {
		idata = NODE_NUMDATA(n);
		for (i=0; i<n->N; i++)
			if (idata[i] == data)
				return npast + i;
		npast += n->N;
	}
	return -1;
}

int NLF(contains)(nl* list, const number data) {
	return (NLF(index_of)(list, data) != -1);
}

int NLF(sorted_contains)(nl* list, const number n) {
	return NLF(sorted_index_of)(list, n) != -1;
}

int NLF(sorted_index_of)(nl* list, const number n) {
	bl_node *node;
	int lower;
	int nskipped;

	node = NLF(findnodecontainingsorted)(list, n, &nskipped);
	if (!node)
		return -1;
	//if (!node && (n > NODE_NUMDATA(prev)[prev->N-1]))
	//return -1;
	//node = prev;

	 /*
	 // find the first node for which n <= the last element.  That node
	 // will contain the value (if it exists)
	 for (; node && (n > NODE_NUMDATA(node)[node->N-1]); node=node->next)
	 nskipped += node->N;
	 if (!node)
	 return -1;
	 */

	// update jump accessors...
	list->last_access = node;
	list->last_access_n = nskipped;

	// find within the node...
	lower = NLF(binarysearch)(node, n);
	if (lower >= 0 && n == NODE_NUMDATA(node)[lower])
		return nskipped + lower;
	return -1;
}

#undef NLF
#undef NODE_NUMDATA
