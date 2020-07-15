/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AN_SCAMP_H
#define AN_SCAMP_H

#include "astrometry/qfits_header.h"
#include "astrometry/sip.h"
#include "astrometry/starxy.h"

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
