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

#ifndef VERIFY_H
#define VERIFY_H

#include "kdtree.h"
#include "matchobj.h"
#include "bl.h"
#include "starkd.h"
#include "sip.h"
#include "bl.h"

struct verify_field_t {
    double* field;
    double* fieldcopy;
    int NF;
    kdtree_t* ftree;
};
typedef struct verify_field_t verify_field_t;

void verify_hit(startree_t* skdt,
                MatchObj* mo,
                sip_t* sip, // if non-NULL, verify this SIP WCS.
                verify_field_t* vf,
                double verify_pix2,
                double distractors,
                double fieldW,
                double fieldH,
                double logratio_tobail,
                bool distance_from_quad_bonus,
				int dimquads,
                bool fake_match);

verify_field_t* verify_field_preprocess(double* field, int NF);

void verify_field_free(verify_field_t* vf);

void verify_init();

void verify_cleanup();

#endif
