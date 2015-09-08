/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#ifndef VERIFY2_H
#define VERIFY2_H

#include "astrometry/kdtree.h"
#include "astrometry/matchobj.h"
#include "astrometry/bl.h"
#include "astrometry/starkd.h"
#include "astrometry/sip.h"
#include "astrometry/bl.h"
#include "astrometry/starxy.h"
#include "astrometry/index.h"

void verify_get_all_matches(const double* refxys, int NR,
							const double* testxys, const double* testsigma2s, int NT,
							double effective_area,
							double distractors,
							double nsigma,
							double limit,
							il*** p_reflist,
							dl*** p_problist);

#endif
