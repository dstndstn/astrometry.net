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

#ifndef idfile_H
#define idfile_H

#include <sys/types.h>
#include <stdint.h>
#include <assert.h>

#include "qfits.h"
#include "fitsbin.h"

struct idfile {
	unsigned int numstars;
	int healpix;

    fitsbin_t* fb;

	// when reading:
	uint64_t* anidarray;
};
typedef struct idfile idfile;

idfile* idfile_open(char* fn);

idfile* idfile_open_for_writing(char* fn);

qfits_header* idfile_get_header(idfile* id);

int idfile_close(idfile* id);

uint64_t idfile_get_anid(idfile* id, unsigned int starid);

int idfile_write_anid(idfile* id, uint64_t anid /* astrometry.net id */ );

int idfile_fix_header(idfile* id);

int idfile_write_header(idfile* id);


#endif
