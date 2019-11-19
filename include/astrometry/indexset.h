/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef AN_INDEXSET_H
#define AN_INDEXSET_H

#include "astrometry/index.h"
#include "astrometry/an-bool.h"
#include "astrometry/anqfits.h"
#include "astrometry/bl.h"

void indexset_get(const char* name, pl* indexlist);

#endif
