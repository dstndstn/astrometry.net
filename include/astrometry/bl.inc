/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2009 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include <assert.h>

#include "astrometry/bl.ph"

/* find the node in which element "n" can be found. */
InlineDefine
bl_node* find_node(const bl* list, size_t n,
				   size_t* p_nskipped) {
	bl_node* node;
	size_t nskipped;
	if (list->last_access && n >= list->last_access_n) {
		// take the shortcut!
		nskipped = list->last_access_n;
		node = list->last_access;
	} else {
		node = list->head;
		nskipped = 0;
	}

	for (; node;) {
		if (n < (nskipped + node->N))
			break;
		nskipped += node->N;
		node = node->next;
	}

	assert(node);

	if (p_nskipped)
		*p_nskipped = nskipped;

	return node;
}

InlineDefine
void* bl_access(bl* list, size_t n) {
	void* rtn;
	bl_node* node;
	size_t nskipped;
	node = find_node(list, n, &nskipped);
	// grab the element.
	rtn = NODE_CHARDATA(node) + (n - nskipped) * list->datasize;
	// update the last_access member...
	list->last_access = node;
	list->last_access_n = nskipped;
	return rtn;
}

InlineDefine
size_t bl_size(const bl* list) {
	return list->N;
}


InlineDefine
size_t sl_size(const sl* list) {
	return bl_size(list);
}

