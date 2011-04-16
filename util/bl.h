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

/**
   A linked list of arrays, which allows
   more rapid traversal of the list, and fairly
   efficient insertions and deletions.
*/

#ifndef BL_H
#define BL_H

#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>

#include "keywords.h"

struct bl_node {
	// number of elements filled.
	int N;
	struct bl_node* next;
	// (data block implicitly follows this struct).
};
typedef struct bl_node bl_node;

// the top-level data structure of a blocklist.
struct bl {
  bl_node* head;
  bl_node* tail;
	// the total number of data elements
  int N;
	// the number of elements per block
  int blocksize;
	// the size in bytes of each data element
  int datasize;
	// rapid accessors for "jumping in" at the last block accessed
  bl_node* last_access;
  int last_access_n;
};
typedef struct bl bl;


Malloc bl*  bl_new(int blocksize, int datasize);
void bl_init(bl* l, int blocksize, int datasize);
void bl_free(bl* list);
void  bl_remove_all(bl* list);
Pure InlineDeclare int  bl_size(const bl* list);
Pure int  bl_datasize(const bl* list);
/** Appends an element, returning the location whereto it was copied. */
void* bl_append(bl* list, const void* data);
// Copies the nth element into the destination location.
void  bl_get(bl* list, int n, void* dest);
// Returns a pointer to the nth element.
InlineDeclare void* bl_access(bl* list, int n);

void* bl_access_const(const bl* list, int n);

void* bl_push(bl* list, const void* data);
// Pops a data item into the given "into" memory.
void  bl_pop(bl* list, void* into);

// allocates space for a new object and returns a pointer to it
void* bl_extend(bl* list);

/**
   Removes elements from \c split
   to the end of the list from \c src and appends them to \c dest.
 */
void bl_split(bl* src, bl* dest, int split);

void bl_reverse(bl* list);

/*
 * Appends "list2" to the end of "list1", and removes all elements
 * from "list2".
 */
void bl_append_list(bl* list1, bl* list2);
void bl_insert(bl* list, int indx, const void* data);
void bl_set(bl* list, int indx, const void* data);
/**
 * Inserts the given datum into the list in such a way that the list
 * stays sorted in ascending order according to the given comparison
 * function (assuming it was sorted to begin with!).
 *
 * The inserted element will be placed _after_ existing elements with
 * the same value.
 *
 * The comparison function is the same as qsort's: it should return
 * 1 if the first arg is greater than the second arg
 * 0 if they're equal
 * -1 if the first arg is smaller.
 *
 * The index where the element was inserted is returned.
 */
int bl_insert_sorted(bl* list, const void* data, int (*compare)(const void* v1, const void* v2));

/**
   If the item already existed in the list (ie, the compare function
   returned zero), then -1 is returned.  Otherwise, the index at which
   the item was inserted is returned.
 */
int bl_insert_unique_sorted(bl* list, const void* data,
                            int (*compare)(const void* v1, const void* v2));

void bl_sort(bl* list, int (*compare)(const void* v1, const void* v2));

void  bl_print_structure(bl* list);
void  bl_copy(bl* list, int start, int length, void* vdest);
/*
  Removes all the elements, but doesn't free the first block, which makes
  it slightly faster for the case when you're going to add more elements
  right away, since you don't have to free() the old block then immediately
  malloc() a new block.
*/
void  bl_remove_all_but_first(bl* list);
void  bl_remove_index(bl* list, int indx);
void  bl_remove_index_range(bl* list, int start, int length);
void* bl_find(bl* list, const void* data, int (*compare)(const void* v1, const void* v2));
int   bl_find_index(bl* list, const void* data, int (*compare)(const void* v1, const void* v2));

// returns 0 if okay, 1 if an error is detected.
int   bl_check_consistency(bl* list);

// returns 0 if okay, 1 if an error is detected.
int   bl_check_sorted(bl* list, int (*compare)(const void* v1, const void* v2), int isunique);

///////////////////////////////////////////////
// special-case functions for pointer lists. //
///////////////////////////////////////////////
//pl*   pl_dup(pl* list);
//#define pl_clear pl_remove_all

///////////////////////////////////////////////
// special-case functions for string lists.  //
///////////////////////////////////////////////
/*
  sl makes a copy of the string using strdup().
  It will be freed when the string is removed from the list or the list is
  freed.
*/
typedef bl sl;
sl*    sl_new(int blocksize);

/*
 The functions:
   sl_init()  --->  sl_init2()
   sl_free()  --->  sl_free2()
   sl_add()   --->  sl_add2()
   sl_find()  --->  sl_find2()
 are defined by BSD, where they live in libc.

 We therefore avoid these names, which breaks the principle of least surprise, but
 makes life a bit easier.
 */

void   sl_init2(sl* list, int blocksize);

// free this list and all the strings it contains.
void   sl_free2(sl* list);

void sl_append_contents(sl* dest, sl* src);

// Searches the sl for the given string.  Comparisons use strcmp().
// Returns -1 if the string is not found, or the first index where it was found.
int sl_index_of(sl* lst, const char* str);
int sl_last_index_of(sl* lst, const char* str);

// Returns 0 if the string is not in the sl, 1 otherwise.
// (same as sl_index_of(lst, str) > -1)
int sl_contains(sl* lst, const char* str);

// just free the list structure, not the strings in it.
void   sl_free_nonrecursive(sl* list);

Pure InlineDeclare int  sl_size(const sl* list);

// copies the string and enqueues it; returns the newly-allocate string.
char*  sl_append(sl* list, const char* string);
// appends the string; doesn't copy it.
void   sl_append_nocopy(sl* list, const char* string);

void sl_append_array(sl* list, const char** strings, int n);

// copies the string and pushes the copy.  Returns the copy.
char*  sl_push(sl* list, const char* data);
// returns the last string: it's your responsibility to free it.
char*  sl_pop(sl* list);
char*  sl_get(sl* list, int n);
char*  sl_get_const(const sl* list, int n);
// sets the string at the given index to the given value.
// if there is already a string at that index, frees it.
char*  sl_set(sl* list, int n, const char* val);
int    sl_check_consistency(sl* list);
// inserts a copy of the given string.
char*  sl_insert(sl* list, int indx, const char* str);
// inserts the given string.
void sl_insert_nocopy(sl* list, int indx, const char* str);
// frees all the strings and removes them from the list.
void   sl_remove_all(sl* list);

// inserts the string; doesn't copy it.
void   sl_insert_sorted_nocopy(sl* list, const char* string);

// inserts a copy of the string; returns the newly-allocated string.
char* sl_insert_sorted(sl* list, const char* string);

void sl_remove_index_range(sl* list, int start, int length);

void sl_remove(sl* list, int index);

// Removes "string" if it is found in the list.
// Note that this checks pointer match, not a strcmp() match.
// Returns the index where the string was found, or -1 if it wasn't found.
int sl_remove_string(sl* list, const char* string);

// Removes "string" if it is found in the list, using strcasecmp().
// Returns the string or NULL if not found.
char* sl_remove_string_bycaseval(sl* list, const char* string);

// Removes "string" if it is found in the list, using strcmp().
// Returns the index where the string was found, or -1 if it wasn't found.
int sl_remove_string_byval(sl* list, const char* string);

// remove all elements starting from "start" to the end of the list.
void sl_remove_from(sl* list, int start);

void   sl_merge_lists(sl* list1, sl* list2);
void   sl_print(sl* list);

/*
 Removes duplicate entries, using strcmp().
 */
void sl_remove_duplicates(sl* lst);

/*
 Splits the given string 'str' into substrings separated by 'sepstring'.
 (Some of the substrings may be empty, for example if the 'sepstring' appears
 consecutively.)
 Adds them to 'lst', if non-NULL.
 Allocates and fills a new sl* if 'lst' is NULL.
 The original string can be reconstructed by calling "sl_implode(lst, sepstring)"
 */
sl* sl_split(sl* lst, const char* str, const char* sepstring);

// Like the PHP function implode(), joins each element in the list with the given
// "join" string.  The result is a newly-allocate string containing:
//   sl_get(list, 0) + join + sl_get(list, 1) + join + ... + join + sl_get(list, N-1)
// -AKA sl_join.
char*  sl_implode(sl* list, const char* join);

// like Python's joinstring.join(list)
// -AKA sl_implode
char*  sl_join(sl* list, const char* joinstring);

// same as sl_join(reverse(list), str)
char*  sl_join_reverse(sl* list, const char* join);

// Appends the (newly-allocated) formatted string and returns it.
char*
ATTRIB_FORMAT(printf,2,3)
sl_appendf(sl* list, const char* format, ...);

// Appends the (newly-allocated) formatted string and returns it.
char* sl_appendvf(sl* list, const char* format, va_list va);

// Inserts the (newly-allocated) formatted string and returns it.
char*
ATTRIB_FORMAT(printf,2,3)
sl_insert_sortedf(sl* list, const char* format, ...);

// Inserts the (newly-allocated) formatted string and returns it.
char*
ATTRIB_FORMAT(printf,3,4)
sl_insertf(sl* list, int index, const char* format, ...);


#define DEFINE_SORT 1
#define nl il
#define number int
#include "bl-nl.h"
#undef nl
#undef number

#define nl ll
#define number int64_t
#include "bl-nl.h"
#undef nl
#undef number

#define nl dl
#define number double
#include "bl-nl.h"
#undef nl
#undef number

#define nl fl
#define number float
#include "bl-nl.h"
#undef nl
#undef number

#undef DEFINE_SORT
#define DEFINE_SORT 0
#define nl pl
#define number void*
#include "bl-nl.h"
#undef nl
#undef number

#undef DEFINE_SORT

//////// Special functions ////////
void  pl_free_elements(pl* list);
void  pl_sort(pl* list, int (*compare)(const void* v1, const void* v2));
int pl_insert_sorted(pl* list, const void* data, int (*compare)(const void* v1, const void* v2));


#ifdef INCLUDE_INLINE_SOURCE
#define InlineDefine InlineDefineH

#include "bl.inc"

#define nl il
#define number int
#include "bl-nl.inc"
#undef nl
#undef number

#define nl ll
#define number int64_t
#include "bl-nl.inc"
#undef nl
#undef number

#define nl pl
#define number void*
#include "bl-nl.inc"
#undef nl
#undef number

#define nl dl
#define number double
#include "bl-nl.inc"
#undef nl
#undef number

#define nl fl
#define number float
#include "bl-nl.inc"
#undef nl
#undef number

#undef InlineDefine
#endif


#endif
