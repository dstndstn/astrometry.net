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

#include <assert.h>

#include "intmap.h"

int intmap_get(intmap* map, int from, int fail) {
	int indx = il_index_of(&map->fromlist, from);
	if (indx == -1) return fail;
	return il_get(&map->tolist, indx);
}

int intmap_get_inverse(intmap* map, int to, int fail) {
	int indx = il_index_of(&map->tolist, to);
	if (indx == -1) return fail;
	return il_get(&map->fromlist, indx);
}

void intmap_get_entry(intmap* map, int indx, int* pfrom, int* pto) {
	int from, to;
	assert(indx < il_size(&map->fromlist));
	if (pfrom) {
		from = il_get(&map->fromlist, indx);
		*pfrom = from;
	}
	if (pto) {
		to = il_get(&map->tolist, indx);
		*pto = to;
	}
}

int intmap_length(intmap* m) {
	return il_size(&m->fromlist);
}

intmap* intmap_new(intmap_type type) {
	intmap* c = malloc(sizeof(intmap));
	intmap_init(c, type);
	return c;
}

void intmap_init(intmap* c, intmap_type type) {
	il_new_existing(&c->fromlist, 4);
	il_new_existing(&c->tolist, 4);
	c->type = type;
}

void intmap_clear(intmap* m) {
	il_remove_all(&m->fromlist);
	il_remove_all(&m->tolist);
}

void intmap_free(intmap* map) {
	intmap_clear(map);
	free(map);
}

int intmap_conflicts(intmap* c, int from, int to) {
	int i, len;
	len = il_size(&c->fromlist);
	for (i=0; i<len; i++) {
		int f, t;
		f = il_get(&c->fromlist, i);
		t = il_get(&c->tolist  , i);
		if ((from == f) && (to == t)) {
			// okay.
			continue;
		}
		switch (c->type) {
		case INTMAP_ONE_TO_ONE:
			if ((from == f) || (to == t))
				// conflict!
				return 1;
		case INTMAP_MANY_TO_ONE:
			if (from == f)
				// conflict!
				return 1;
		}
	}
	return 0;
}

int intmap_add(intmap* c, int from, int to) {
	int i, len;
	len = il_size(&c->fromlist);
	for (i=0; i<len; i++) {
		int f, t;
		f = il_get(&c->fromlist, i);
		t = il_get(&c->tolist  , i);
		if ((from == f) && (to == t)) {
			// mapping exists.
			return 1;
		}
		switch (c->type) {
		case INTMAP_ONE_TO_ONE:
			if ((from == f) || (to == t))
				// conflict!
				return -1;
		case INTMAP_MANY_TO_ONE:
			if (from == f)
				// conflict!
				return -1;
		}
	}
	il_append(&c->fromlist, from);
	il_append(&c->tolist, to);
	return 0;
}

void intmap_update(intmap* map, int from, int to) {
	int i, len;
	len = il_size(&map->fromlist);
	for (i=0; i<len; i++) {
		int f, t;
		f = il_get(&map->fromlist, i);
		t = il_get(&map->tolist  , i);
		if ((from == f) && (to == t)) {
			// mapping exists.
			return;
		}
		switch (map->type) {
		case INTMAP_ONE_TO_ONE:
			if ((from == f) || (to == t)) {
				il_set(&map->fromlist, i, from);
				il_set(&map->tolist, i, to);
				return;
			}
			break;
		case INTMAP_MANY_TO_ONE:
			assert(0);  // Why knows what this is supposed to do?
			exit(-1);
			/*
			  if (from == f) {
			  il_set(&map->fromlist, i, from);
			  il_set(&map->tolist, i, to);
			  return;
			  }
			*/
		}
	}
	il_append(&map->fromlist, from);
	il_append(&map->tolist, to);
}

int intmap_merge(intmap* map1, intmap* map2) {
	int i, len;
	int okay = 1;
	len = il_size(&map2->fromlist);
	for (i=0; i<len; i++) {
		int from, to;
		from = il_get(&map2->fromlist, i);
		to   = il_get(&map2->tolist  , i);
		if (intmap_add(map1, from, to) == -1) {
			okay = 0;
		}
	}
	if (okay)
		return 0;
	return -1;
}
