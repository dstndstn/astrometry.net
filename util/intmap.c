/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2012 Dustin Lang.

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
#include <assert.h>

#include "intmap.h"

#define IMGLUE2(n,f) n ## _ ## f
#define IMGLUE(n,f) IMGLUE2(n,f)

#define key_t int
#define nl il
#define KL(x) IMGLUE(nl, x)

intmap_t* intmap_new(int datasize, int subblocksize, int blocksize,
					 int Ndense) {
	intmap_t* im = calloc(1, sizeof(intmap_t));
	if (!blocksize)
		blocksize = 4096;
	im->blocksize = subblocksize;
	im->datasize = datasize;
	if (Ndense) {
		im->ND = Ndense;
		im->dense = calloc(im->ND, sizeof(bl*));
	} else {
		im->keys = KL(new)(blocksize);
		im->lists = pl_new(blocksize);
	}
	return im;
}

void intmap_free(intmap_t* im) {
	int i;
	if (im->lists) {
		for (i=0; i<pl_size(im->lists); i++) {
			bl* lst = pl_get(im->lists, i);
			bl_free(lst);
		}
		pl_free(im->lists);
	}
	if (im->dense) {
		for (i=0; i<im->ND; i++) {
			bl* lst = im->dense[i];
			if (!lst)
				continue;
			bl_free(lst);
		}
		free(im->dense);
	}
	if (im->keys)
		KL(free)(im->keys);
	free(im);
}

bl* intmap_find(intmap_t* im, key_t key, anbool create) {
	key_t ind;
	assert(key >= 0);
	assert(im);
	if (!im->dense) {
		assert(im->keys);
		assert(im->lists);
		ind = KL(sorted_index_of)(im->keys, key);
		if (ind == -1) {
			bl* lst;
			if (!create)
				return NULL;
			lst = bl_new(im->blocksize, im->datasize);
			ind = KL(insert_unique_ascending)(im->keys, key);
			pl_insert(im->lists, ind, lst);
			return lst;
		}
		return pl_get(im->lists, ind);
	} else {
		bl* lst;
		assert(key < im->ND);
		assert(im->dense);
		lst = im->dense[key];
		if (lst)
			return lst;
		if (!create)
			return lst;
		lst = im->dense[key] = bl_new(im->blocksize, im->datasize);
		return lst;
	}
}

void intmap_append(intmap_t* it, int key, void* pval) {
	bl* lst = intmap_find(it, key, TRUE);
	bl_append(lst, pval);
}

anbool intmap_get_entry(intmap_t* im, int index,
					  key_t* p_key, bl** p_list) {
	assert(im);
	assert(index >= 0);
	if (im->dense) {
		if (index >= im->ND)
			return FALSE;
		if (p_key)
			*p_key = index;
		if (p_list)
			*p_list = im->dense[index];
		return TRUE;
	}

	assert(im->keys);
	assert(im->lists);
	if (index >= KL(size)(im->keys))
		return FALSE;
	if (p_key)
		*p_key = KL(get)(im->keys, index);
	if (p_list)
		*p_list = pl_get(im->lists, index);
	return TRUE;
}

#undef IMGLUE2
#undef IMGLUE
#undef key
#undef nl
#undef KL
