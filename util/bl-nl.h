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
 Common header for lists of numerical types.

 Expects "nl" to be #defined to the list type.

 Expects "number" to be #defined to the numerical type.
 */

#include "bl-nl.ph"

typedef bl nl;

// The "const number"s in here are mostly for pl.

Malloc nl*  NLF(new)(int blocksize);
Pure InlineDeclare int NLF(size)(const nl* list);
void NLF(new_existing)(nl* list, int blocksize);
void NLF(init)(nl* list, int blocksize);
void NLF(reverse)(nl* list);
void NLF(remove_all)(nl* list);
void NLF(remove_all_reuse)(nl* list);
void NLF(free)(nl* list);
number* NLF(append)(nl* list, const number data);
void NLF(append_list)(nl* list, nl* list2);
void NLF(append_array)(nl* list, const number* data, int ndata);
void NLF(merge_lists)(nl* list1, nl* list2);
void NLF(push)(nl* list, const number data);
number  NLF(pop)(nl* list);
int NLF(contains)(nl* list, const number data);
// Assuming the list is sorted in ascending order,
// does it contain the given number?
int NLF(sorted_contains)(nl* list, const number data);
// Or -1 if not found.
int NLF(sorted_index_of)(nl* list, const number data);

#if DEFINE_SORT
void NLF(sort)(nl* list, int ascending);
#endif

Malloc number* NLF(to_array)(nl* list);

// Returns the index in the list of the given number, or -1 if it
// is not found.
int  NLF(index_of)(nl* list, const number data);
InlineDeclare number  NLF(get)(nl* list, int n);
InlineDeclare number  NLF(get_const)(const nl* list, int n);

InlineDeclare number*  NLF(access)(nl* list, int n);

/**
 Copy from the list, starting at index "start" for length "length",
 into the provided array.
 */
void NLF(copy)(nl* list, int start, int length, number* vdest);
nl*  NLF(dupe)(nl* list);
void NLF(print)(nl* list);
void   NLF(insert)(nl* list, int indx, const number data);
int NLF(insert_ascending)(nl* list, const number n);
int NLF(insert_descending)(nl* list, const number n);
// Returns the index at which the element was added, or -1 if it's a duplicate.
int  NLF(insert_unique_ascending)(nl* list, const number p);
void NLF(set)(nl* list, int ind, const number value);
void NLF(remove)(nl* list, int ind);
void NLF(remove_index_range)(nl* list, int start, int length);
// See also sorted_index_of, which should be faster.
int  NLF(find_index_ascending)(nl* list, const number value);

nl* NLF(merge_ascending)(nl* list1, nl* list2);

// returns the index of the removed value, or -1 if it didn't
// exist in the list.
int NLF(remove_value)(nl* list, const number value);

int NLF(check_consistency)(nl* list);
int NLF(check_sorted_ascending)(nl* list, int isunique);
int NLF(check_sorted_descending)(nl* list, int isunique);


