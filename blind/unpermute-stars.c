/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "kdtree.h"
#include "starutil.h"
#include "quadfile.h"
#include "fitsioutils.h"
#include "qfits.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"

int unpermute_stars(const char* skdtinfn, const char* quadinfn,
					const char* skdtoutfn, const char* quadoutfn,
					bool dosweeps, bool check,
					char** args, int argc) {
    quadfile* qfin;
	quadfile* qfout;
	startree_t* treein;
	startree_t* treeout;
	int i;
	int N;
	int healpix;
    int hpnside;
	int starhp;
	int lastgrass;
	qfits_header* qouthdr;
	qfits_header* qinhdr;

	logmsg("Reading star tree from %s ...\n", skdtinfn);
	treein = startree_open(skdtinfn);
	if (!treein) {
		ERROR("Failed to read star kdtree from %s.\n", skdtinfn);
		return -1;
	}
	N = startree_N(treein);

	logmsg("Reading quadfile from %s ...\n", quadinfn);
	qfin = quadfile_open(quadinfn);
	if (!qfin) {
		ERROR("Failed to read quadfile from %s.\n", quadinfn);
		return -1;
	}

	starhp = qfits_header_getint(startree_header(treein), "HEALPIX", -1);
	if (starhp == -1)
		ERROR("Warning, input star kdtree didn't have a HEALPIX header.\n");
    healpix = starhp;
	hpnside = qfits_header_getint(startree_header(treein), "HPNSIDE", 1);

	logmsg("Writing quadfile to %s ...\n", quadoutfn);
	qfout = quadfile_open_for_writing(quadoutfn);
	if (!qfout) {
		ERROR("Failed to write quadfile to %s.\n", quadoutfn);
		return -1;
	}

	qfout->healpix = healpix;
    qfout->hpnside = hpnside;
	qfout->numstars          = qfin->numstars;
	qfout->dimquads          = qfin->dimquads;
	qfout->index_scale_upper = qfin->index_scale_upper;
	qfout->index_scale_lower = qfin->index_scale_lower;
	qfout->indexid           = qfin->indexid;

	qouthdr = quadfile_get_header(qfout);
	qinhdr  = quadfile_get_header(qfin);

	boilerplate_add_fits_headers(qouthdr);
	qfits_header_add(qouthdr, "HISTORY", "This file was created by the program \"unpermute-stars\".", NULL, NULL);
	qfits_header_add(qouthdr, "HISTORY", "unpermute-stars command line:", NULL, NULL);
	fits_add_args(qouthdr, args, argc);
	qfits_header_add(qouthdr, "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
	qfits_header_add(qouthdr, "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
	fits_copy_all_headers(qinhdr, qouthdr, "HISTORY");
	qfits_header_add(qouthdr, "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
	qfits_header_add(qouthdr, "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
	fits_copy_all_headers(qinhdr, qouthdr, "COMMENT");
	qfits_header_add(qouthdr, "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);

	if (quadfile_write_header(qfout)) {
		ERROR("Failed to write quadfile header.\n");
		return -1;
	}

	logmsg("Writing quads...\n");

	startree_compute_inverse_perm(treein);

	if (check) {
		logmsg("Running quadfile_check()...\n");
		if (quadfile_check(qfin)) {
			ERROR("quadfile_check() failed");
			return -1;
		}
		logmsg("Check passed.\n");

		logmsg("Checking inverse permutation...\n");
		if (startree_check_inverse_perm(treein)) {
			ERROR("check failed!");
			return -1;
		}

		logmsg("Running startree kdtree_check()...\n");
		if (kdtree_check(treein->tree)) {
			ERROR("kdtree_check() failed");
			return -1;
		}
		logmsg("Check passed.\n");
	}


	lastgrass = 0;
	for (i=0; i<qfin->numquads; i++) {
		int j;
		unsigned int stars[qfin->dimquads];
		if (i*80/qfin->numquads != lastgrass) {
			logmsg(".");
			fflush(stdout);
			lastgrass = i*80/qfin->numquads;
		}
		if (quadfile_get_stars(qfin, i, stars)) {
			ERROR("Failed to read quadfile entry.\n");
			return -1;
        }
		for (j=0; j<qfin->dimquads; j++)
			stars[j] = treein->inverse_perm[stars[j]];
		if (quadfile_write_quad(qfout, stars)) {
			ERROR("Failed to write quadfile entry.\n");
			return -1;
		}
	}
	logmsg("\n");

	quadfile_close(qfin);

	if (quadfile_fix_header(qfout) ||
		quadfile_close(qfout)) {
		ERROR("Failed to close output quadfile.\n");
		return -1;
	}

	treeout = startree_new();
	treeout->tree = malloc(sizeof(kdtree_t));
	memcpy(treeout->tree, treein->tree, sizeof(kdtree_t));
	treeout->tree->perm = NULL;

	fits_copy_header(startree_header(treein), startree_header(treeout), "HEALPIX");
	fits_copy_header(startree_header(treein), startree_header(treeout), "HPNSIDE");
	fits_copy_header(startree_header(treein), startree_header(treeout), "ALLSKY");
	fits_copy_header(startree_header(treein), startree_header(treeout), "JITTER");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTNSIDE");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTMARG");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTBAND");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTDEDUP");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTNSWEP");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTMINMG");
	fits_copy_header(startree_header(treein), startree_header(treeout), "CUTMAXMG");

	qfits_header_add(startree_header(treeout), "HISTORY", "unpermute-stars command line:", NULL, NULL);
	fits_add_args(startree_header(treeout), args, argc);
	qfits_header_add(startree_header(treeout), "HISTORY", "(end of unpermute-stars command line)", NULL, NULL);
	qfits_header_add(startree_header(treeout), "HISTORY", "** unpermute-stars: history from input:", NULL, NULL);
	fits_copy_all_headers(startree_header(treein), startree_header(treeout), "HISTORY");
	qfits_header_add(startree_header(treeout), "HISTORY", "** unpermute-stars: end of history from input.", NULL, NULL);
	qfits_header_add(startree_header(treeout), "COMMENT", "** unpermute-stars: comments from input:", NULL, NULL);
	fits_copy_all_headers(startree_header(treein), startree_header(treeout), "COMMENT");
	qfits_header_add(startree_header(treeout), "COMMENT", "** unpermute-stars: end of comments from input.", NULL, NULL);

	if (dosweeps) {
		// copy sweepX headers.
		for (i=1;; i++) {
			char key[16];
			int n;
			sprintf(key, "SWEEP%i", i);
			n = qfits_header_getint(treein->header, key, -1);
			if (n == -1)
				break;
			fits_copy_header(treein->header, treeout->header, key);
		}

		// compute sweep array.
		treeout->sweep = malloc(N * sizeof(uint8_t));
		for (i=0; i<N; i++) {
			int ind = treein->tree->perm[i];
			// Stars are sorted first by sweep and then by brightness within
			// the sweep.  Instead of just storing the sweep number, we can
			// store a quantization of the total-ordered rank.
			treeout->sweep[i] = (uint8_t)(256 * ind / N);
		}
	}

    logmsg("Permuting tag-along arrays...\n");
    if (treein->sigma_radec)
        treeout->sigma_radec   = malloc(N * 2 * sizeof(float));
    if (treein->proper_motion)
        treeout->proper_motion = malloc(N * 2 * sizeof(float));
    if (treein->sigma_pm)
        treeout->sigma_pm      = malloc(N * 2 * sizeof(float));
    if (treein->starids)
        treeout->starids       = malloc(N * sizeof(uint64_t));

    for (i=0; i<N; i++) {
        int ind = treein->tree->perm[i];
        if (treein->sigma_radec) {
            treeout->sigma_radec[2*i] = treein->sigma_radec[2*ind];
            treeout->sigma_radec[2*i+1] = treein->sigma_radec[2*ind+1];
        }
        if (treein->proper_motion) {
            treeout->proper_motion[2*i] = treein->proper_motion[ind];
            treeout->proper_motion[2*i+1] = treein->proper_motion[2*ind+1];
        }
        if (treein->sigma_pm) {
            treeout->sigma_pm[2*i] = treein->sigma_pm[2*ind];
            treeout->sigma_pm[2*i+1] = treein->sigma_pm[2*ind+1];
        }
        if (treein->starids)
            treeout->starids[i] = treein->starids[ind];
    }

	logmsg("Writing star kdtree to %s ...\n", skdtoutfn);
	if (startree_write_to_file(treeout, skdtoutfn)) {
		ERROR("Failed to write star kdtree.\n");
		return -1;
	}

    free(treeout->sigma_radec);
    free(treeout->proper_motion);
    free(treeout->sigma_pm);
    free(treeout->starids);

	startree_close(treein);
	free(treeout->sweep);
    free(treeout->tree);
    treeout->tree = NULL;
	startree_close(treeout);

	return 0;
}
