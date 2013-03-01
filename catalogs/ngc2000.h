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

/**
 The Astrometry.net codebase has two NGC modules.  This one contains
 rough positions for all NGC/IC objects.  ngcic-accurate.h contains
 more precise positions for some of the objects.

 You probably want to use them something like this:

 int i, N;
 
 N = ngc_num_entries();
 for (i=0; i<N; i++) {
   ngc_entry* ngc = ngc_get_entry_accurate(i);
   // do stuff ...
   // (do NOT free(ngc); !)
 }


 */

/*
  The NGC2000 catalog can be found at:
    ftp://cdsarc.u-strasbg.fr/cats/VII/118/

  The "ReadMe" file associated with the catalog is ngc2000-readme.txt
*/

struct ngc_entry {
	// true: NGC.  false: IC.
	anbool is_ngc;

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
	// anbool sizelimit;
	float mag;
	// anbool photo_mag;
	// char[51] description;
};
typedef struct ngc_entry ngc_entry;

extern ngc_entry ngc_entries[];

// convenience accessors:

// Find an entry by NGC/IC number.
ngc_entry* ngc_get_ngcic_num(anbool is_ngc, int num);

int ngc_num_entries();

ngc_entry* ngc_get_entry(int i);

ngc_entry* ngc_get_entry_named(const char* name);

// Checks the "ngcic-accurate" catalog for more accurate RA,Dec
// and substitutes it if found.
ngc_entry* ngc_get_entry_accurate(int i);

// find the common name of the given ngc_entry, if it has one.
char* ngc_get_name(ngc_entry* entry, int num);

// Returns "NGC ###" or "IC ###" plus the common names.
// The names will be added to the given "lst" if it is supplied.
// A new list will be created if "lst" is NULL.
sl* ngc_get_names(ngc_entry* entry, sl* lst);

char* ngc_get_name_list(ngc_entry* entry, const char* separator);

#endif
