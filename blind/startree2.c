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

#include "startree2.h"
#include "kdtree.h"
#include "errors.h"
#include "log.h"
#include "starutil.h"
#include "an-bool.h"
#include "fitsioutils.h"
#include "boilerplate.h"
#include "fitstable.h"

anbool startree_has_tagalong_data(const fitstable_t* intab) {
	// don't include RA,Dec.
	return fitstable_n_fits_columns(intab) > 2;
}

int startree_write_tagalong_table(fitstable_t* intab, fitstable_t* outtab,
								  const char* racol, const char* deccol) {
	int i, R, NB, N;
	char* buf;
	qfits_header* hdr;
	
	fitstable_clear_table(intab);
	fitstable_add_fits_columns_as_struct(intab);
	fitstable_copy_columns(intab, outtab);
	if (!racol)
		racol = "RA";
	if (!deccol)
		deccol = "DEC";
	fitstable_remove_column(outtab, racol);
	fitstable_remove_column(outtab, deccol);
    fitstable_read_extension(intab, 1);
	hdr = fitstable_get_header(outtab);
	qfits_header_add(hdr, "AN_FILE", AN_FILETYPE_TAGALONG, "Extra data for stars", NULL);
	if (fitstable_write_header(outtab)) {
		ERROR("Failed to write tag-along data header");
		return -1;
	}
	R = fitstable_row_size(intab);
	NB = 1000;
	logverb("Input row size: %i, output row size: %i\n", R, fitstable_row_size(outtab));
	buf = malloc(NB * R);
	N = fitstable_nrows(intab);
	
	for (i=0; i<N; i+=NB) {
		int nr = NB;
		if (i+NB > N)
			nr = N - i;
		if (fitstable_read_structs(intab, buf, R, i, nr)) {
			ERROR("Failed to read tag-along data from catalog");
			return -1;
		}
		if (fitstable_write_structs(outtab, buf, R, nr)) {
			ERROR("Failed to write tag-along data");
			return -1;
		}
	}
	free(buf);
	if (fitstable_fix_header(outtab)) {
		ERROR("Failed to fix tag-along data header");
		return -1;
	}
	return 0;
}

startree_t* startree_build(fitstable_t* intable,
						   const char* racol, const char* deccol,
						   // keep RA,Dec in the tag-along table?
						   //anbool keep_radec,
						   // KDT_DATA_*, KDT_TREE_*
						   int datatype, int treetype,
						   // KD_BUILD_*
						   int buildopts,
						   int Nleaf,
						   char** args, int argc) {
	double* ra = NULL;
	double* dec = NULL;
	double* xyz = NULL;
	int N;
	startree_t* starkd = NULL;
	int tt;
	int d;
	double low[3];
	double high[3];
	qfits_header* hdr;
	qfits_header* inhdr;
	int i;

	if (!racol)
		racol = "RA";
	if (!deccol)
		deccol = "DEC";
	if (!datatype)
		datatype = KDT_DATA_U32;
	if (!treetype)
		treetype = KDT_TREE_U32;
	if (!buildopts)
		buildopts = KD_BUILD_SPLIT;
	if (!Nleaf)
		Nleaf = 25;


	ra = fitstable_read_column(intable, racol, TFITS_BIN_TYPE_D);
	if (!ra) {
		ERROR("Failed to read RA from column %s", racol);
		goto bailout;
	}
	dec = fitstable_read_column(intable, deccol, TFITS_BIN_TYPE_D);
	if (!dec) {
		ERROR("Failed to read RA from column %s", racol);
		goto bailout;
	}
	N = fitstable_nrows(intable);
	xyz = malloc(N * 3 * sizeof(double));
	if (!xyz) {
		SYSERROR("Failed to malloc xyz array to build startree");
		goto bailout;
	}
	radecdeg2xyzarrmany(ra, dec, xyz, N);
	free(ra);
	ra = NULL;
	free(dec);
	dec = NULL;

	starkd = startree_new();
	if (!starkd) {
		ERROR("Failed to allocate startree");
		goto bailout;
	}
	tt = kdtree_kdtypes_to_treetype(KDT_EXT_DOUBLE, treetype, datatype);
	starkd->tree = kdtree_new(N, 3, Nleaf);
	for (d=0; d<3; d++) {
		low[d] = -1.0;
		high[d] = 1.0;
	}
	kdtree_set_limits(starkd->tree, low, high);
	logverb("Building star kdtree...\n");
	starkd->tree = kdtree_build(starkd->tree, xyz, N, 3, Nleaf, tt, buildopts);
	if (!starkd->tree) {
		ERROR("Failed to build star kdtree");
		startree_close(starkd);
		starkd = NULL;
		goto bailout;
	}
	starkd->tree->name = strdup(STARTREE_NAME);

	inhdr = fitstable_get_primary_header(intable);
    hdr = startree_header(starkd);
	fits_copy_header(inhdr, hdr, "HEALPIX");
	fits_copy_header(inhdr, hdr, "HPNSIDE");
	fits_copy_header(inhdr, hdr, "ALLSKY");
	fits_copy_header(inhdr, hdr, "JITTER");
	fits_copy_header(inhdr, hdr, "CUTNSIDE");
	fits_copy_header(inhdr, hdr, "CUTMARG");
	fits_copy_header(inhdr, hdr, "CUTDEDUP");
	fits_copy_header(inhdr, hdr, "CUTNSWEP");
	//fits_copy_header(inhdr, hdr, "CUTBAND");
	//fits_copy_header(inhdr, hdr, "CUTMINMG");
	//fits_copy_header(inhdr, hdr, "CUTMAXMG");
	boilerplate_add_fits_headers(hdr);
	qfits_header_add(hdr, "HISTORY", "This file was created by the command-line:", NULL, NULL);
	fits_add_args(hdr, args, argc);
	qfits_header_add(hdr, "HISTORY", "(end of command line)", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "** History entries copied from the input file:", NULL, NULL);
	fits_copy_all_headers(inhdr, hdr, "HISTORY");
	qfits_header_add(hdr, "HISTORY", "** End of history entries.", NULL, NULL);
	for (i=1;; i++) {
		char key[16];
		int n;
		sprintf(key, "SWEEP%i", i);
		n = qfits_header_getint(inhdr, key, -1);
		if (n == -1)
			break;
		fits_copy_header(inhdr, hdr, key);
	}

 bailout:
	if (ra)
		free(ra);
	if (dec)
		free(dec);
	if (xyz)
		free(xyz);
	return starkd;
}

