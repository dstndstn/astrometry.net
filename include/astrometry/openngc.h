/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef OPENNGC_H
#define OPENNGC_H

#include "astrometry/bl.h"

/**
 The Astrometry.net codebase contains an NGC module based on the
 OpenNGC catalog, which can be found at:
   https://github.com/mattiaverga/OpenNGC/

 You probably want to use it something like this:

 int i, N;

 N = ngc_num_entries();
 for (i=0; i<N; i++) {
   ngc_entry* ngc = ngc_get_entry(i);
   // do stuff ...
   // (do NOT free(ngc); !)
 }

 */

struct ngc_entry {
    // true: NGC.  false: IC.
    int is_ngc;

    // NGC/IC number
    int id;

    // RA,Dec in J2000.0 degrees
    float ra;
    float dec;

    // Maximum dimension in arcmin.
    float size;
};
typedef struct ngc_entry ngc_entry;

extern ngc_entry ngc_entries[];

// convenience accessors:

// Find an entry by NGC/IC number.
ngc_entry* ngc_get_ngcic_num(int is_ngc, int num);

int ngc_num_entries();

ngc_entry* ngc_get_entry(int i);

ngc_entry* ngc_get_entry_named(const char* name);

// find the common name of the given ngc_entry, if it has one.
char* ngc_get_name(ngc_entry* entry, int num);

// Returns "NGC ###" or "IC ###" plus the common names.
// The names will be added to the given "lst" if it is supplied.
// A new list will be created if "lst" is NULL.
sl* ngc_get_names(ngc_entry* entry, sl* lst);

char* ngc_get_name_list(ngc_entry* entry, const char* separator);

#endif
