/*
  This file is part of the Astrometry.net suite.
  Copyright 2010 Dustin Lang.

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

#include <sys/param.h>

#include "sip.h"
#include "matchobj.h"
#include "starutil.h"
#include "log.h"
#include "verify.h"

void matchobj_compute_derived(MatchObj* mo) {
	int mx;
	int i;
	mx = 0;
	for (i=0; i<mo->dimquads; i++)
		mx = MAX(mx, mo->field[i]);
	mo->objs_tried = mx+1;
	if (mo->wcs_valid)
		mo->scale = tan_pixel_scale(&(mo->wcstan));
    mo->radius = deg2dist(mo->radius_deg);
	mo->nbest = mo->nmatch + mo->ndistractor + mo->nconflict;
}

/*void matchobj_log_verify_hit_miss(MatchObj* mo, int loglevel) {
 verify_log_hit_miss(mo->theta, mo->nbest, mo->nfield, loglevel);
 }
 */
