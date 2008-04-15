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

#ifndef INTMAP_H
#define INTMAP_H

#include "bl.h"

enum intmap_type {
	INTMAP_ONE_TO_ONE,
	INTMAP_MANY_TO_ONE
};
typedef enum intmap_type intmap_type;

struct intmap {
	il fromlist;
	il tolist;
	intmap_type type;
};
typedef struct intmap intmap;

int intmap_length(intmap* m);

intmap* intmap_new(intmap_type type);

void intmap_init(intmap* m, intmap_type type);

void intmap_clear(intmap* m);

void intmap_free(intmap* map);

int intmap_conflicts(intmap* map, int from, int to);

// returns 0 if it was added okay;
//         1 if the mapping already existed.
//        -1 if a conflict would have been created.
int intmap_add(intmap* map, int from, int to);

// like _add but overwrites existing.
void intmap_update(intmap* map, int from, int to);

// adds all elements in "map2" into "map1".
// returns 0 if it worked okay, -1 if a conflict would have been created.
int intmap_merge(intmap* map1, intmap* map2);

void intmap_get_entry(intmap* map, int indx, int* pfrom, int* pto);

// look up the value "from" in the mapping; return "fail" if it's not found.
int intmap_get(intmap* map, int from, int fail);

int intmap_get_inverse(intmap* map, int to, int fail);

#endif
