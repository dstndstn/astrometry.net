/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
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

#ifndef WCS_RD2XY_H
#define WCS_RD2XY_H

#include "bl.h"

int wcs_rd2xy(const char* wcsfn, int wcsext,
			  const char* rdlsfn, const char* xylsfn,
              const char* racol, const char* deccol,
			  anbool forcetan,
			  anbool forcewcslib,
              il* fields);

#endif
