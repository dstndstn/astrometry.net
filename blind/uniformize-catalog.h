/*
  This file is part of the Astrometry.net suite.
  Copyright 2009, 2011 Dustin Lang.

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
#ifndef UNIFORMIZE_CATALOG_H
#define UNIFORMIZE_CATALOG_H

#include "fitstable.h"
#include "an-bool.h"

/**

 Given a FITS BINTABLE, select a (permuted) subset of the rows that
 yields a spatially uniform, bright, catalog.

 Works by laying a healpix grid over the sky and selecting N stars in
 each healpixel (grid cell).  In other words, sweep over the
 healpixes, first selecting the brightest star, then the second
 brightest, etc.

 healpix = -1: make all-sky index.


 FIXME -- the following is currently NOT TRUE -- do we need it?
 Within each sweep, the brightness ordering will be
 used, so that when you list the catalog you will see the stars
 collected during the first sweep, sorted by brightness, then the
 stars from the second sweep, also sorted by brightness.  Note that
 the brightest star in the second sweep may be brighter than the
 faintest star in the first sweep: spatial uniformity trumps
 brightness.


 */
int uniformize_catalog(fitstable_t* intable, fitstable_t* outtable,
					   const char* racol, const char* deccol,
					   // ? Or do this sorting in a separate step?
					   const char* sortcol, anbool sort_ascending,
					   double sort_min_cut,
					   // ?  Or do this cut in a separate step?
					   int healpix, int hpnside,
					   int nmargin,
					   // uniformization nside.
					   int finenside,
					   double dedup_radius_arcsec,
					   int nsweeps,
					   char** args, int argc);

#endif
