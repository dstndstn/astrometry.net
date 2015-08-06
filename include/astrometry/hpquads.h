/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2, or
  (at your option) any later version.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef HPQUADS_H
#define HPQUADS_H

#include "astrometry/an-bool.h"
#include "astrometry/starkd.h"
#include "astrometry/codefile.h"
#include "astrometry/quadfile.h"

int hpquads(startree_t* starkd,
			codefile_t* codes,
			quadfile_t* quads,
			int Nside,
			double scale_min_arcmin,
			double scale_max_arcmin,
			int dimquads,
			int passes,
			int Nreuses,
			int Nloosen,
			int id,
			anbool scanoccupied,

			void* sort_data,
			int (*sort_func)(const void*, const void*),
			int sort_size,

			char** args, int argc);

int hpquads_files(const char* skdtfn,
				  const char* codefn,
				  const char* quadfn,
				  int Nside,
				  double scale_min_arcmin,
				  double scale_max_arcmin,
				  int dimquads,
				  int passes,
				  int Nreuses,
				  int Nloosen,
				  int id,
				  anbool scanoccupied,

				  void* sort_data,
				  int (*sort_func)(const void*, const void*),
				  int sort_size,

				  char** args, int argc);

#endif



