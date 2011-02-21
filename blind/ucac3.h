/*
  This file is part of the Astrometry.net suite.
  Copyright 2011 Dustin Lang.

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

#ifndef UCAC3_H
#define UCAC3_H

#include "an-bool.h"
#include <stdint.h>
#include "starutil.h"

#define UCAC3_RECORD_SIZE 84

struct ucac3_entry {
	// (in brackets are the name, format, and units in the UCAC3 data files)
	// [degrees] (ra, I4: mas)
	double ra;
	// [degrees] (spd, I4: mas)
	double dec;

	// [degrees] (sigra, I2, mas)
	// error in RA*cos(Dec)
	float sigma_ra;
	// [degrees] (sigdc, I2, mas)
	float sigma_dec;

	// fit model mag
	// [mag] (im1, I2: millimag)
	float mag;

	// aperture mag
	// [mag] (im2, I2: millimag)
	float apmag;

	// [mag] (sigmag, I2: millimag)
	float sigmag;

	// (objt, I1)
	int8_t objtype;

	// (dsf, I1)
	int8_t doublestar;

	// (na1: I1)
	// total # of CCD images of this star
	uint8_t na1;

	// (nu1: I1)
	// # of CCD images used for this star
	uint8_t nu1;

	// (us1: I1)
	// # catalogs (epochs) used for proper motions
	uint8_t us1;

	// (cn1: I1)
	// total numb. catalogs (epochs) initial match
	uint8_t cn1;

	// Central epoch for mean RA/Dec
	// [yr] (cepra/cepdc, I2, 0.01 yr - 1900)
	float epoch_ra;
	float epoch_dec;

	// Proper motion in RA*cos(Dec), Dec
	// [arcsec/yr] (pmrac/pmdc, I4, 0.1 mas/yr)
	float pm_ra;
	float pm_dec;

	// [arcsec/pr] (sigpmr/sigpmd, I2, 0.1 mas/yr)
	float sigma_pm_ra;
	float sigma_pm_dec;

	// 2MASS pts_key star identifier
	// (id2m, I4)
	uint32_t twomass_id;

	// 2MASS J mag
	// (jmag, I2, millimag)
	float jmag;

	// 2MASS H mag
	// (hmag, I2, millimag)
	float hmag;

	// 2MASS K_s mag
	// (kmag, I2, millimag)
	float kmag;

	// e2mpho I*1 * 3         2MASS error photom. (1/100 mag)
	float jmag_err;
	float hmag_err;
	float kmag_err;

	// icqflg I*1 * 3         2MASS cc_flg*10 + phot.qual.flag
	uint8_t twomass_jflags;
	uint8_t twomass_hflags;
	uint8_t twomass_kflags;

	// SuperCosmos Bmag / R2mag / Imag
	// (smB / smR2 / smI, I2 millimag)
	float bmag;
	float r2mag;
	float imag;
	// clbl   I*1             SC star/galaxy classif./quality flag
	uint8_t clbl;

	// SuperCosmos quality flag Bmag/R2mag/Imag
	// (qfB/qfR2/qfI, I1)
	uint8_t bquality;
	uint8_t r2quality;
	uint8_t iquality;

	// mmf flag for 10 major catalogs matched
	// (catflg, I * 10)
	uint8_t mmf[10];

	// Yale SPM object type (g-flag)
	// (g1, I1)
	uint8_t g1;
	// Yale SPM input cat.  (c-flag)
	// (c1, I1)
	uint8_t c1;

	// LEDA galaxy match flag
	// (leda, I1)
	uint8_t leda_flag;

	// 2MASS extend.source flag
	// (x2m, I1)
	uint8_t twomass_extsource_flag;

	// MPOS star number; identifies HPM stars
	// (rn, I4)
};

typedef struct ucac3_entry ucac3_entry;

#endif
