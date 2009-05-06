/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2009 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include "quad-utils.h"
#include "starutil.h"
#include "codefile.h"
#include "starkd.h"
#include "errors.h"

int quad_compute_code(const unsigned int* quad, int dimquads, startree_t* starkd, 
					  double* code) {
    int i;
	double starxyz[3 * DQMAX];
	for (i=0; i<dimquads; i++) {
		if (startree_get(starkd, quad[i], starxyz + 3*i)) {
			ERROR("Failed to get stars belonging to a quad.\n");
			return -1;
		}
	}
    codefile_compute_star_code(starxyz, code, dimquads);
	return 0;
}

void quad_enforce_invariants(unsigned int* quad, double* code,
							 int dimquads, int dimcodes) {
	double sum;
	int i;

	// here we add the invariant that (cx + dx + ...) / (dimquads-2) <= 1/2
	sum = 0.0;
	for (i=0; i<(dimquads-2); i++)
		sum += code[2*i];
	sum /= (dimquads-2);
	if (sum > 0.5) {
		// swap the labels of A,B.
		int tmp = quad[0];
		quad[0] = quad[1];
		quad[1] = tmp;
		// rotate the code 180 degrees.
		for (i=0; i<dimcodes; i++)
			code[i] = 1.0 - code[i];
	}

	// here we add the invariant that cx <= dx <= ....
	for (i=0; i<(dimquads-2); i++) {
		int j;
		int jsmallest;
		double smallest;
		double x1;
		double dtmp;
		int tmp;

		x1 = code[2*i];
		jsmallest = -1;
		smallest = x1;
		for (j=i+1; j<(dimquads-2); j++) {
			double x2 = code[2*j];
			if (x2 < smallest) {
				smallest = x2;
				jsmallest = j;
			}
		}
		if (jsmallest == -1)
			continue;
		j = jsmallest;
		// swap the labels.
		tmp = quad[i+2];
		quad[i+2] = quad[j+2];
		quad[j+2] = tmp;
		// swap the code values.
		dtmp = code[2*i];
		code[2*i] = code[2*j];
		code[2*j] = dtmp;
		dtmp = code[2*i+1];
		code[2*i+1] = code[2*j+1];
		code[2*j+1] = dtmp;
	}
}

void quad_write(codefile* codes, quadfile* quads,
				unsigned int* quad, startree_t* starkd,
				int dimquads, int dimcodes) {
	double code[DCMAX];

	quad_compute_code(quad, dimquads, starkd, code);
	quad_enforce_invariants(quad, code, dimquads, dimcodes);
	codefile_write_code(codes, code);
	quadfile_write_quad(quads, quad);
}

