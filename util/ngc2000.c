/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ngc2000.h"
#include "bl.h"

struct ngc_name {
	bool is_ngc;
	int id;
	char* name;
};
typedef struct ngc_name ngc_name;

ngc_name ngc_names[] = {
#include "ngc2000names.c"
};

ngc_entry ngc_entries[] = {
#include "ngc2000entries.c"
};

int ngc_num_entries() {
	return sizeof(ngc_entries) / sizeof(ngc_entry);
}

ngc_entry* ngc_get_entry(int i) {
	if (i < 0)
		return NULL;
	if (i >= (sizeof(ngc_entries) / sizeof(ngc_entry)))
		return NULL;
	return ngc_entries + i;
}

char* ngc_get_name(ngc_entry* entry, int num) {
	int i;
	for (i=0; i<sizeof(ngc_names)/sizeof(ngc_name); i++) {
		if ((entry->is_ngc == ngc_names[i].is_ngc) &&
			(entry->id == ngc_names[i].id)) {
			if (num == 0)
				return ngc_names[i].name;
			else
				num--;
		}
	}
	return NULL;
}

sl* ngc_get_names(ngc_entry* entry) {
	int i;
	sl* list = NULL;
	for (i=0; i<sizeof(ngc_names)/sizeof(ngc_name); i++) {
		if ((entry->is_ngc == ngc_names[i].is_ngc) &&
			(entry->id == ngc_names[i].id)) {
			if (!list)
				list = sl_new(4);
			sl_append(list, ngc_names[i].name);
		}
	}
	return list;
}
