/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef INTMAP_H
#define INTMAP_H

#include "astrometry/an-bool.h"
#include "astrometry/bl.h"

/**
 Maps integers to lists of objects.
 */


// FIXME -- would this work better with "bt" for the keys?
struct intmap {
	// dense only:
	bl** dense;
	int ND;

	// sparse only:
	il* keys;
	// list of bl*
	pl* lists;

	// common:
	// list blocksize
	int blocksize;
	// data size
	int datasize;
};
typedef struct intmap intmap_t;

/**
 Creates a new intmap that will contain data objects (ie, list
 elements) of size "datasize".  The lists will have blocks of size
 "subblocksize".  Intmap uses blocklists internally, and their block
 sizes are set by "blocksize".

 "Ndense": if non-zero, the number of items that will be contained.
 If zero, a sparse map is assumed.
 */
intmap_t* intmap_new(int datasize, int subblocksize, int blocksize,
					 int Ndense);

void intmap_free(intmap_t* it);

/**
 Finds the list of objects for the given "key".  Creates a new list if
 "create" is TRUE and the list didn't already exist.
 */
bl* intmap_find(intmap_t* it, int key, anbool create);

void intmap_append(intmap_t* it, int key, void* pval);

/**
 Iterates through the map elements.

 Returns pointers to the "index"-th entry.

 Returns TRUE if the element exists, FALSE otherwise.

 The iteration proceeds in a random order.
 */
anbool intmap_get_entry(intmap_t* it, int index, int* key, bl** list);

#endif

