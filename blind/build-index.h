/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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
#ifndef BUILD_INDEX_H
#define BUILD_INDEX_H

#include "fitstable.h"
#include "index.h"
#include "an-bool.h"
#include "index.h"

struct index_params {
	// catalog:
	const char* racol;
	const char* deccol;
	// in arcsec
	double jitter;

	// uniformization:
	const char* sortcol;
	anbool sortasc;

	double brightcut;
	int bighp;
	int bignside;
	int sweeps;
	double dedup;
	int margin;
	int UNside;

	// hpquads:
	int Nside;

	void* hpquads_sort_data;
	int (*hpquads_sort_func)(const void*, const void*);
	int hpquads_sort_size;

	// quad size range, in arcmin
	double qlo; double qhi;
	int passes;
	int Nreuse; int Nloosen;
	anbool scanoccupied;
	int dimquads;
	int indexid;

	// general options
	anbool inmemory;
	anbool delete_tempfiles;
	char* tempdir;
	char** args;
	int argc;
};
typedef struct index_params index_params_t;

void build_index_defaults(index_params_t* params);

int build_index_files(const char* catalogfn, const char* indexfn,
					  index_params_t* params);

int build_index(fitstable_t* catalog, index_params_t* p,
				index_t** p_index, const char* indexfn);

int build_index_shared_skdt(const char* starkdfn, startree_t* starkd,
							index_params_t* p,
							index_t** p_index, const char* indexfn);

int build_index_shared_skdt_files(const char* starkdfn, const char* indexfn,
								  index_params_t* p);

#endif
