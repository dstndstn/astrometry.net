/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2009, 2012 Dustin Lang.

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

/**
   Reads a list of codes and writes a code kdtree.

   Input: .code
   Output: .ckdt
*/
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "codetree.h"
#include "codefile.h"
#include "fitsioutils.h"
#include "codekd.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"

int codetree_files(const char* codefn, const char* ckdtfn,
				   int Nleaf, int datatype, int treetype,
				   int buildopts,
				   char** args, int argc) {
    codefile_t* codes;
    codetree_t *codekd = NULL;

	assert(codefn);
	assert(ckdtfn);
    logmsg("codetree: building KD tree for %s\n", codefn);
    logmsg("       will write KD tree file %s\n", ckdtfn);
    logmsg("Reading codes...\n");

    codes = codefile_open(codefn);
    if (!codes) {
		ERROR("Failed to read code file %s", codefn);
		return -1;
    }
    logmsg("Read %u codes.\n", codes->numcodes);

	codekd = codetree_build(codes, Nleaf, datatype, treetype,
							buildopts, args, argc);
	if (!codekd) {
		return -1;
	}

    logmsg("Writing code KD tree to %s...\n", ckdtfn);
	if (codetree_write_to_file(codekd, ckdtfn)) {
        ERROR("Failed to write code kdtree to %s", ckdtfn);
		return -1;
    }
    codefile_close(codes);
    kdtree_free(codekd->tree);
    codekd->tree = NULL;
    codetree_close(codekd);
	return 0;
}

codetree_t* codetree_build(codefile_t* codes,
						 int Nleaf, int datatype, int treetype,
						 int buildopts,
						 char** args, int argc) {
	codetree_t* codekd;
	qfits_header* hdr;
	int exttype = KDT_EXT_DOUBLE;
	int tt;
	int N, D;
    qfits_header* chdr;

	codekd = codetree_new();
	if (!codekd) {
		ERROR("Failed to allocate a codetree structure");
		return NULL;
	}

	if (!Nleaf)
		Nleaf = 25;
	if (!datatype)
		datatype = KDT_DATA_U16;
	if (!treetype)
		treetype = KDT_TREE_U16;
	if (!buildopts)
		buildopts = KD_BUILD_SPLIT;

	tt = kdtree_kdtypes_to_treetype(exttype, treetype, datatype);
	N = codes->numcodes;
	D = codefile_dimcodes(codes);
	codekd->tree = kdtree_new(N, D, Nleaf);
    chdr = codefile_get_header(codes);
	{
		double low[D];
		double high[D];
		int d;
		anbool circ;
        circ = qfits_header_getboolean(chdr, "CIRCLE", 0);
		for (d=0; d<D; d++) {
			if (circ) {
				low [d] = 0.5 - M_SQRT1_2;
				high[d] = 0.5 + M_SQRT1_2;
			} else {
				low [d] = 0.0;
				high[d] = 1.0;
			}
		}
		kdtree_set_limits(codekd->tree, low, high);
	}
    logmsg("Building tree...\n");
    codekd->tree = kdtree_build(codekd->tree, codes->codearray, N, D,
                                Nleaf, tt, buildopts);
    if (!codekd->tree) {
		ERROR("Failed to build code kdtree");
		return NULL;
	}
    logmsg("Done\n");
    codekd->tree->name = strdup(CODETREE_NAME);

	hdr = codetree_header(codekd);
	fits_header_add_int(hdr, "NLEAF", Nleaf, "Target number of points in leaves.");
	an_fits_copy_header(chdr, hdr, "INDEXID");
	an_fits_copy_header(chdr, hdr, "HEALPIX");
	an_fits_copy_header(chdr, hdr, "ALLSKY");
	an_fits_copy_header(chdr, hdr, "HPNSIDE");
	an_fits_copy_header(chdr, hdr, "CXDX");
	an_fits_copy_header(chdr, hdr, "CXDXLT1");
	an_fits_copy_header(chdr, hdr, "CIRCLE");
	BOILERPLATE_ADD_FITS_HEADERS(hdr);
	qfits_header_add(hdr, "HISTORY", "This file was created by the command-line:", NULL, NULL);
	fits_add_args(hdr, args, argc);
	qfits_header_add(hdr, "HISTORY", "(end of command line)", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "** codetree: history from input file:", NULL, NULL);
	fits_copy_all_headers(chdr, hdr, "HISTORY");
	qfits_header_add(hdr, "HISTORY", "** codetree: end of history from input file.", NULL, NULL);

	return codekd;
}

