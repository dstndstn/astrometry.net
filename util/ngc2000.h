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

#ifndef NGC2000_H
#define NGC2000_H

#include "an-bool.h"
#include "bl.h"

/*
  The NGC2000 catalog can be found at:
    ftp://cdsarc.u-strasbg.fr/cats/VII/118/

  The "ReadMe" file associated with the catalog is ngc2000-readme.txt
*/

struct ngc_entry {
	// true: NGC.  false: IC.
	bool is_ngc;

	// NGC/IC number
	int id;

	char classification[4];

	// RA,Dec in B2000.0 degrees
	float ra;
	float dec;

	char constellation[4];

	// Maximum dimension in arcmin.
	float size;

	//char source;
	// bool sizelimit;
	// float mag;
	// bool photo_mag;
	// char[51] description;
};
typedef struct ngc_entry ngc_entry;

// find the common of the given ngc_entry, if it has one.
char* ngc_get_name(ngc_entry* entry, int num);

sl* ngc_get_names(ngc_entry* entry);

extern ngc_entry ngc_entries[];

// convenience accessors:

int ngc_num_entries();

ngc_entry* ngc_get_entry(int i);

#endif
