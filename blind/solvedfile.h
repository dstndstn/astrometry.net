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

#ifndef SOLVEDFILE_H
#define SOLVEDFILE_H

#include "bl.h"

int solvedfile_get(char* fn, int fieldnum);

int solvedfile_getsize(char* fn);

/**
   Get a list of unsolved fields between "firstfield" and
   "lastfield", up to a maximum of "maxfields" (no limit if
   "maxfields" is zero).
 */
il* solvedfile_getall(char* fn, int firstfield, int lastfield, int maxfields);

/**
   Same as "getall" except return solved fields.
 */
il* solvedfile_getall_solved(char* fn, int firstfield, int lastfield, int maxfields);

int solvedfile_set(char* fn, int fieldnum);

int solvedfile_setsize(char* fn, int fieldnum);

#endif
