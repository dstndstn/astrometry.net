/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

/**
 Common header for lists of numerical types.

 Expects "nl" to be #defined to the list type.

 Expects "number" to be #defined to the numerical type.
 */

#include "astrometry/bl-nl.ph"

typedef bl nl;

// The "const number"s in here are mostly for pl.

Malloc nl*  NLF(new)(int blocksize);
Pure InlineDeclare size_t NLF(size)(const nl* list);
void NLF(new_existing)(nl* list, int blocksize);
void NLF(init)(nl* list, int blocksize);
void NLF(reverse)(nl* list);
void NLF(remove_all)(nl* list);
void NLF(remove_all_reuse)(nl* list);
void NLF(free)(nl* list);
number* NLF(append)(nl* list, const number data);
void NLF(append_list)(nl* list, nl* list2);
void NLF(append_array)(nl* list, const number* data, size_t ndata);
void NLF(merge_lists)(nl* list1, nl* list2);
void NLF(push)(nl* list, const number data);
number  NLF(pop)(nl* list);
int NLF(contains)(nl* list, const number data);
// Assuming the list is sorted in ascending order,
// does it contain the given number?
int NLF(sorted_contains)(nl* list, const number data);
// Or -1 if not found.
ptrdiff_t NLF(sorted_index_of)(nl* list, const number data);

#if DEFINE_SORT
void NLF(sort)(nl* list, int ascending);
#endif

Malloc number* NLF(to_array)(nl* list);

// Returns the index in the list of the given number, or -1 if it
// is not found.
ptrdiff_t  NLF(index_of)(nl* list, const number data);
InlineDeclare number  NLF(get)(nl* list, size_t n);
InlineDeclare number  NLF(get_const)(const nl* list, size_t n);

InlineDeclare number*  NLF(access)(nl* list, size_t n);

/**
 Copy from the list, starting at index "start" for length "length",
 into the provided array.
 */
void NLF(copy)(nl* list, size_t start, size_t length, number* vdest);
nl*  NLF(dupe)(nl* list);
void NLF(print)(nl* list);
void   NLF(insert)(nl* list, size_t indx, const number data);
size_t NLF(insert_ascending)(nl* list, const number n);
size_t NLF(insert_descending)(nl* list, const number n);
// Returns the index at which the element was added, or -1 if it's a duplicate.
ptrdiff_t  NLF(insert_unique_ascending)(nl* list, const number p);
void NLF(set)(nl* list, size_t ind, const number value);
void NLF(remove)(nl* list, size_t ind);
void NLF(remove_index_range)(nl* list, size_t start, size_t length);
// See also sorted_index_of, which should be faster.
// Or -1 if not found
ptrdiff_t  NLF(find_index_ascending)(nl* list, const number value);

nl* NLF(merge_ascending)(nl* list1, nl* list2);

// returns the index of the removed value, or -1 if it didn't
// exist in the list.
ptrdiff_t NLF(remove_value)(nl* list, const number value);

int NLF(check_consistency)(nl* list);
int NLF(check_sorted_ascending)(nl* list, int isunique);
int NLF(check_sorted_descending)(nl* list, int isunique);


