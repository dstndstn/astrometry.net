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

#ifndef ALLQUADS_H
#define ALLQUADS_H

struct allquads {
	int dimquads;
	int dimcodes;
	int id;

	char *quadfn;
	char *codefn;
	char *skdtfn;

	startree_t* starkd;
	quadfile* quads;
	codefile* codes;

	double quad_d2_lower;
	double quad_d2_upper;
	anbool use_d2_lower;
	anbool use_d2_upper;

	int starA;
};
typedef struct allquads allquads_t;

allquads_t* allquads_init();
int allquads_open_outputs(allquads_t* aq);
int allquads_create_quads(allquads_t* aq);
int allquads_close(allquads_t* aq);
void allquads_free(allquads_t* aq);


#endif

