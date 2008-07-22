/*
 This file is part of the Astrometry.net suite.
 Copyright 2008 Dustin Lang.

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

#ifndef BLIND_SCAMP_H
#define BLIND_SCAMP_H

#include "qfits.h"
#include "sip.h"
#include "starxy.h"

int scamp_write_field(const qfits_header* imageheader,
                      const sip_t* wcs,
                      const starxy_t* xy,
                      const char* filename);


// Writes a Scamp config file snippet describing the reference and input catalogs
// we generate.
int scamp_write_config_file(const char* refcatfn, const char* outfn);

// Returns a newly-allocated Scamp config file snippet describing the reference
// and input catalogs we generate.
char* scamp_get_config_options(const char* refcatfn);

#endif
