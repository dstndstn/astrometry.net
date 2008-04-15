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

#include <stdint.h>

#include "nomad.h"
#include "an-endian.h"
#include "starutil.h"

int nomad_parse_entry(nomad_entry* entry, const void* encoded) {
	const uint32_t* udata = encoded;
	uint32_t uval;
	int32_t ival;

	ival = uval = u32_letoh(udata[0]);
	entry->ra = arcsec2deg(uval * 0.001);

	ival = uval = u32_letoh(udata[1]);
	entry->dec = arcsec2deg(uval * 0.001) - 90.0;
    
	ival = uval = u32_letoh(udata[2]);
	entry->sigma_racosdec = arcsec2deg(uval * 0.001);

	ival = uval = u32_letoh(udata[3]);
	entry->sigma_dec = arcsec2deg(uval * 0.001);

	ival = uval = u32_letoh(udata[4]);
	entry->pm_racosdec = ival * 0.0001;

	ival = uval = u32_letoh(udata[5]);
	entry->pm_dec = ival * 0.0001;

	ival = uval = u32_letoh(udata[6]);
	entry->sigma_pm_racosdec = uval * 0.0001;

	ival = uval = u32_letoh(udata[7]);
	entry->sigma_pm_dec = uval * 0.0001;

	ival = uval = u32_letoh(udata[8]);
	entry->epoch_ra = uval * 0.001;

	ival = uval = u32_letoh(udata[9]);
	entry->epoch_dec = uval * 0.001;

	ival = uval = u32_letoh(udata[10]);
	entry->mag_B = ival * 0.001;

	ival = uval = u32_letoh(udata[11]);
	entry->mag_V = ival * 0.001;

	ival = uval = u32_letoh(udata[12]);
	entry->mag_R = ival * 0.001;

	ival = uval = u32_letoh(udata[13]);
	entry->mag_J = ival * 0.001;

	ival = uval = u32_letoh(udata[14]);
	entry->mag_H = ival * 0.001;

	ival = uval = u32_letoh(udata[15]);
	entry->mag_K = ival * 0.001;

	ival = uval = u32_letoh(udata[16]);
	entry->usnob_id = uval;

	ival = uval = u32_letoh(udata[17]);
	entry->twomass_id = uval;

	ival = uval = u32_letoh(udata[18]);
	entry->yb6_id = uval;

	ival = uval = u32_letoh(udata[19]);
	entry->ucac2_id = uval;

	ival = uval = u32_letoh(udata[20]);
	entry->tycho2_id = uval;

	ival = uval = u32_letoh(udata[21]);
	entry->astrometry_src = (uval >> 0) & 0x7;
	entry->blue_src       = (uval >> 3) & 0x7;
	entry->visual_src     = (uval >> 6) & 0x7;
	entry->red_src        = (uval >> 9) & 0x7;

	entry->usnob_fail       = (uval >> 12) & 0x1;
	entry->twomass_fail     = (uval >> 13) & 0x1;

	entry->tycho_astrometry = (uval >> 16) & 0x1;
	entry->alt_radec        = (uval >> 17) & 0x1;
	//entry->alt_2mass        = (uval >> 18) & 0x1;
	entry->alt_ucac         = (uval >> 19) & 0x1;
	entry->alt_tycho        = (uval >> 20) & 0x1;
	entry->blue_o           = (uval >> 21) & 0x1;
	entry->red_e            = (uval >> 22) & 0x1;
	entry->twomass_only     = (uval >> 23) & 0x1;
	entry->hipp_astrometry  = (uval >> 24) & 0x1;
	entry->diffraction      = (uval >> 25) & 0x1;
	entry->confusion        = (uval >> 26) & 0x1;
	entry->bright_confusion = (uval >> 27) & 0x1;
	entry->bright_artifact  = (uval >> 28) & 0x1;
	entry->standard         = (uval >> 29) & 0x1;
	//entry->external         = (uval >> 30) & 0x1;

	return 0;
}
