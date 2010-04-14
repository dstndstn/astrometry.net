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

#ifndef ANWCSLIB_H
#define ANWCSLIB_H

/** Interface to Mark Calabretta's wcslib, if available. */

#define ANWCS_TYPE_WCSLIB 1


struct anwcs_t {
	/**
	 If type == ANWCS_TYPE_WCSLIB:
	   data is a wcslib  "struct wcsprm*".
	 */
	int type;
	void* data;
};
typedef struct anwcs_t anwcs_t;


anwcs_t* anwcs_open_wcslib(const char* filename, int ext);

int anwcs_radec2pixelxy(const anwcs_t* wcs, double ra, double dec, double* px, double* py);

int anwcs_pixelxy2radec(const anwcs_t* wcs, double px, double py, double* ra, double* dec);

void anwcs_print(const anwcs_t* wcs, FILE* fid);

void anwcs_free(anwcs_t* wcs);


#endif
