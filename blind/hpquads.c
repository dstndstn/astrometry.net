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

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "healpix.h"
#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "tic.h"
#include "fitsioutils.h"
#include "qfits.h"
#include "permutedsort.h"
#include "bt.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "quad-utils.h"
#include "quad-builder.h"

struct hpquads {
	int dimquads;
	int dimcodes;
	int quadsize;
	int Nside;
	int NHP;
	startree_t* starkd;
	quadfile* quads;
	codefile* codes;


	// bounds of quad scale, in distance between AB on the sphere.
	double quad_dist2_upper;
	double quad_dist2_lower;
	// for hp searching
	double radius2;

	bl* quadlist;
	bt* bigquadlist;

	unsigned char* nuses;
	//int Nuses;

	// from find_stars():
	kdtree_qres_t* res;
	int* inds;
	double* stars;
	int Nstars;

	// for create_quad():
	bool quad_created;
	bool count_uses;
	int hp;

	// for build_quads():
	il* retryhps;
};
typedef struct hpquads hpquads_t;


static int compare_quads(const void* v1, const void* v2) {
	const unsigned int* q1 = v1;
	const unsigned int* q2 = v2;
	int i;
	// Hmm... I thought about having a static global "dimquads" here, but
	// instead just ensured that are quad is always initialized to zero so that
	// "star" values between dimquads and DQMAX are always equal.
	for (i=0; i<DQMAX; i++) {
		if (q1[i] > q2[i])
			return 1;
		if (q1[i] < q2[i])
			return -1;
	}
	return 0;
}

static bool find_stars(hpquads_t* me, double radius2, int R) {
	int d, j, N;
	int destind;
	double centre[3];
	int* perm;

	healpix_to_xyzarr(me->hp, me->Nside, 0.5, 0.5, centre);
	me->res = kdtree_rangesearch_options_reuse(me->starkd->tree, me->res,
											   centre, radius2, KD_OPTIONS_RETURN_POINTS);

	// here we could check whether stars are in the box defined by the
	// healpix boundaries plus quad scale, rather than just the circle
	// containing that box.

	N = me->res->nres;
	me->Nstars = N;
	if (N < me->dimquads)
		return FALSE;

	// FIXME -- could merge this step with the sorting step...

	// remove stars that have been used up.
	if (R) {
		destind = 0;
		for (j=0; j<N; j++) {
			if (me->nuses[me->res->inds[j]] >= R)
				continue;
			me->res->inds[destind] = me->res->inds[j];
			for (d=0; d<3; d++)
				me->res->results.d[destind*3+d] = me->res->results.d[j*3+d];
			destind++;
		}
		N = destind;
		if (N < me->dimquads)
			return FALSE;
	}

	// sort the stars in increasing order of index - assume
	// that this corresponds to decreasing order of brightness.

	// find permutation that sorts by index...
	perm = permuted_sort(me->res->inds, sizeof(int), compare_ints_asc, NULL, N);
	// apply the permutation...
	permutation_apply(perm, N, me->res->inds, me->res->inds, sizeof(int));
	permutation_apply(perm, N, me->res->results.d, me->res->results.d, 3 * sizeof(double));
	free(perm);

	me->inds = (int*)me->res->inds;
	me->stars = me->res->results.d;
	me->Nstars = N;

	return TRUE;
}


static bool check_midpoint(quadbuilder_t* qb, pquad_t* pq, void* vtoken) {
	hpquads_t* me = vtoken;
	return (xyzarrtohealpix(pq->midAB, me->Nside) == me->hp);
}

static bool check_full_quad(quadbuilder_t* qb, unsigned int* quad, int nstars, void* vtoken) {
	hpquads_t* me = vtoken;
	bool dup;
	if (!me->bigquadlist)
		return TRUE;
	dup = bt_contains(me->bigquadlist, quad, compare_quads);
	return !dup;
}

static void add_quad(quadbuilder_t* qb, unsigned int* stars, void* vtoken) {
	int i;
	hpquads_t* me = vtoken;
	bl_append(me->quadlist, stars);
	if (me->count_uses) {
		for (i=0; i<me->dimquads; i++)
			me->nuses[stars[i]]++;
	}
	qb->stop_creating = TRUE;
	me->quad_created = TRUE;
}

static bool create_quad(hpquads_t* me, bool count_uses) {
	quadbuilder_t* qb;

	qb = quadbuilder_init();

	qb->starxyz = me->stars;
	qb->starinds = me->inds;
	qb->Nstars = me->Nstars;
	qb->dimquads = me->dimquads;
	qb->quadd2_low = me->quad_dist2_lower;
	qb->quadd2_high = me->quad_dist2_upper;
	qb->check_scale_low = TRUE;
	qb->check_scale_high = TRUE;
	qb->check_AB_stars = check_midpoint;
	qb->check_AB_stars_token = me;
	qb->check_full_quad = check_full_quad;
	qb->check_full_quad_token = me;
	qb->add_quad = add_quad;
	qb->add_quad_token = me;
	me->quad_created = FALSE;
	me->count_uses = count_uses;
	quadbuilder_create(qb);
	quadbuilder_free(qb);

	return me->quad_created;
}


static void add_headers(qfits_header* hdr, char** argv, int argc,
						qfits_header* startreehdr, bool circle,
						int npasses) {
	int i;
	boilerplate_add_fits_headers(hdr);
	qfits_header_add(hdr, "HISTORY", "This file was created by the program \"hpquads\".", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "hpquads command line:", NULL, NULL);
	fits_add_args(hdr, argv, argc);
	qfits_header_add(hdr, "HISTORY", "(end of hpquads command line)", NULL, NULL);

	qfits_header_add(startreehdr, "HISTORY", "** History entries copied from the input file:", NULL, NULL);
	fits_copy_all_headers(startreehdr, hdr, "HISTORY");
	qfits_header_add(startreehdr, "HISTORY", "** End of history entries.", NULL, NULL);

	qfits_header_add(hdr, "CXDX", "T", "All codes have the property cx<=dx.", NULL);
	qfits_header_add(hdr, "CXDXLT1", "T", "All codes have the property cx+dx<=1.", NULL);
	qfits_header_add(hdr, "MIDHALF", "T", "All codes have the property cx+dx<=1.", NULL);

	qfits_header_add(hdr, "CIRCLE", (circle ? "T" : "F"), 
					 (circle ? "Stars C,D live in the circle defined by AB."
					  :        "Stars C,D live in the box defined by AB."), NULL);

	// add placeholders...
	for (i=0; i<npasses; i++) {
		char key[64];
		sprintf(key, "PASS%i", i+1);
		qfits_header_add(hdr, key, "-1", "placeholder", NULL);
	}
}

static int build_quads(hpquads_t* me, int Nhptotry, il* hptotry, int R) {
	int nthispass = 0;
	int lastgrass = 0;
	int i;

	for (i=0; i<Nhptotry; i++) {
		bool ok;
		int hp;
		if ((i * 80 / Nhptotry) != lastgrass) {
			printf(".");
			fflush(stdout);
			lastgrass = i * 80 / Nhptotry;
		}
		if (hptotry)
			hp = il_get(hptotry, i);
		else
			hp = i;
		me->hp = hp;
		me->quad_created = FALSE;
		ok = find_stars(me, me->radius2, R);
		if (ok)
			create_quad(me, TRUE);

		if (me->quad_created)
			nthispass++;
		else {
			if (R && me->Nstars && me->retryhps)
				// there were some stars, and we're counting how many times stars are used.
				//il_insert_unique_ascending(me->retryhps, hp);
				// we don't mind hps showing up multiple times because we want to make up for the lost
				// passes during loosening...
				il_append(me->retryhps, hp);
			// FIXME -- could also track which hps are worth visiting in a future pass
		}
	}
	return nthispass;
}

	
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
				  bool scanoccupied,
				  char** args, int argc) {
	hpquads_t myhpquads;
	hpquads_t* me = &myhpquads;

	int i;
	int pass;
	bool circle = TRUE;
	double radius2;
	il* hptotry;
	int Nhptotry;
	int nquads;
	double hprad;
	double quadscale;

	int skhp, sknside;

	qfits_header* qhdr;
	qfits_header* chdr;

	int N;

	memset(me, 0, sizeof(hpquads_t));

	if (Nside > 13377) {
		// 12 * (13377+1)^2  >  2^31, so healpix arithmetic will fail.
		// This corresponds to about 0.26 arcmin side length -- pretty tiny...
		// Careful use of unsignedness (uint32_t) would bring this to:
		//   Nside = 18918, side length 0.19 arcmin.
		ERROR("Error: maximum healpix Nside = 13377.\n");
		return -1;
	}
	if (Nreuses > 255) {
		ERROR("Error, reuse (-r) must be less than 256.\n");
		return -1;
	}

	me->Nside = Nside;
	me->dimquads = dimquads;
	//me->Nuses = Nreuses;

	me->NHP = 12 * Nside * Nside;
	me->dimcodes = dimquad2dimcode(dimquads);
	me->quadsize = sizeof(unsigned int) * dimquads;

	printf("Nside=%i.  Nside^2=%i.  Number of healpixes=%i.  Healpix side length ~ %g arcmin.\n",
		   me->Nside, me->Nside*me->Nside, me->NHP, healpix_side_length_arcmin(me->Nside));

	tic();
	printf("Reading star kdtree %s ...\n", skdtfn);
	me->starkd = startree_open(skdtfn);
	if (!me->starkd) {
		ERROR("Failed to open star kdtree %s\n", skdtfn);
		return -1;
	}
	N = startree_N(me->starkd);
	printf("Star tree contains %i objects.\n", N);

	// get the "HEALPIX" header from the skdt...
	skhp = qfits_header_getint(startree_header(me->starkd), "HEALPIX", -1);
	if (skhp == -1) {
		if (!qfits_header_getboolean(startree_header(me->starkd), "ALLSKY", FALSE)) {
			logmsg("Warning: skdt does not contain \"HEALPIX\" header.  Code and quad files will not contain this header either.\n");
		}
	}
    // likewise "HPNSIDE"
	sknside = qfits_header_getint(startree_header(me->starkd), "HPNSIDE", 1);

    if (sknside && Nside % sknside) {
        logerr("Error: Nside (-n) must be a multiple of the star kdtree healpixelisation: %i\n", sknside);
		return -1;
    }

	if (!scanoccupied && (N*(skhp == -1 ? 1 : sknside*sknside*12) < me->NHP)) {
		logmsg("\n\n");
		logmsg("NOTE, your star kdtree is sparse (has only a fraction of the stars expected)\n");
		logmsg("  so you probably will get much faster results by setting the \"-E\" command-line\n");
		logmsg("  flag.\n");
		logmsg("\n\n");
	}

	printf("Will write to quad file %s and code file %s\n", quadfn, codefn);

    me->quads = quadfile_open_for_writing(quadfn);
	if (!me->quads) {
		ERROR("Couldn't open file %s to write quads.\n", quadfn);
		return -1;
	}
    me->codes = codefile_open_for_writing(codefn);
	if (!me->codes) {
		ERROR("Couldn't open file %s to write codes.\n", codefn);
		return -1;
	}
	me->quads->dimquads = me->dimquads;
	me->codes->dimcodes = me->dimcodes;
	me->quads->healpix = skhp;
	me->codes->healpix = skhp;
	me->quads->hpnside = sknside;
	me->codes->hpnside = sknside;
	if (id) {
		me->quads->indexid = id;
		me->codes->indexid = id;
	}

	qhdr = quadfile_get_header(me->quads);
	chdr = codefile_get_header(me->codes);

	add_headers(qhdr, args, argc, startree_header(me->starkd), circle, passes);
	add_headers(chdr, args, argc, startree_header(me->starkd), circle, passes);

    if (quadfile_write_header(me->quads)) {
        ERROR("Couldn't write headers to quads file %s\n", quadfn);
		return -1;
    }
    if (codefile_write_header(me->codes)) {
        ERROR("Couldn't write headers to code file %s\n", codefn);
		return -1;
    }

    me->quads->numstars = me->codes->numstars = N;
    me->codes->index_scale_upper = me->quads->index_scale_upper = distsq2rad(me->quad_dist2_upper);
    me->codes->index_scale_lower = me->quads->index_scale_lower = distsq2rad(me->quad_dist2_lower);
	
	me->nuses = calloc(N, sizeof(unsigned char));

	me->quad_dist2_upper = arcmin2distsq(scale_max_arcmin);
	me->quad_dist2_lower = arcmin2distsq(scale_min_arcmin);

	// hprad = sqrt(2) * (healpix side length / 2.)
	hprad = arcmin2dist(healpix_side_length_arcmin(Nside)) * M_SQRT1_2;
	quadscale = 0.5 * sqrt(me->quad_dist2_upper);
	// 1.01 for a bit of safety.  we'll look at a few extra stars.
	radius2 = square(1.01 * (hprad + quadscale));
	me->radius2 = radius2;

	printf("Healpix radius %g arcsec, quad scale %g arcsec, total %g arcsec\n",
		   distsq2arcsec(hprad*hprad),
		   distsq2arcsec(quadscale*quadscale),
		   distsq2arcsec(radius2));

	hptotry = il_new(1024);

	if (scanoccupied) {
		printf("Scanning %i input stars...\n", N);
		for (i=0; i<N; i++) {
			double xyz[3];
			int j;
			if (startree_get(me->starkd, i, xyz)) {
				ERROR("Failed to get star %i", i);
				return -1;
			}
			j = xyzarrtohealpix(xyz, Nside);
			il_insert_unique_ascending(hptotry, j);
		}
		printf("Will check %i healpixes.\n", il_size(hptotry));
	} else {
		if (skhp == -1) {
			// Try all healpixes.
			il_free(hptotry);
			hptotry = NULL;
			Nhptotry = me->NHP;
		} else {
			// The star kdtree may itself be healpixed
			int starhp, starx, stary;
			// In that case, the healpixes we are interested in form a rectangle
			// within a big healpix.  These are the coords (in [0, Nside)) of
			// that rectangle.
			int x0, x1, y0, y1;
			int x, y;
			int nhp;

			healpix_decompose_xy(skhp, &starhp, &starx, &stary, sknside);
			x0 =  starx    * (Nside / sknside);
			x1 = (starx+1) * (Nside / sknside);
			y0 =  stary    * (Nside / sknside);
			y1 = (stary+1) * (Nside / sknside);

			nhp = 0;
			for (y=y0; y<y1; y++) {
				for (x=x0; x<x1; x++) {
					int j = healpix_compose_xy(starhp, x, y, Nside);
					il_append(hptotry, j);
				}
			}
			assert(il_size(hptotry) == (Nside/sknside) * (Nside/sknside));
		}
	}
	if (hptotry)
		Nhptotry = il_size(hptotry);

	me->quadlist = bl_new(65536, me->quadsize);

	if (Nloosen)
		me->retryhps = il_new(1024);

	for (pass=0; pass<passes; pass++) {
		char key[64];
		int nthispass;

		printf("Pass %i of %i.\n", pass+1, passes);
		printf("Trying %i healpixes.\n", Nhptotry);

		nthispass = build_quads(me, Nhptotry, hptotry, Nreuses);

		printf("Made %i quads (out of %i healpixes) this pass.\n", nthispass, Nhptotry);
		printf("Made %i quads so far.\n", (me->bigquadlist ? bt_size(me->bigquadlist) : 0) + bl_size(me->quadlist));

		sprintf(key, "PASS%i", pass+1);
		fits_header_mod_int(chdr, key, nthispass, "quads created in this pass");
		fits_header_mod_int(qhdr, key, nthispass, "quads created in this pass");

		printf("Merging quads...\n");
		if (!me->bigquadlist)
			me->bigquadlist = bt_new(me->quadsize, 256);
		for (i=0; i<bl_size(me->quadlist); i++) {
			void* q = bl_access(me->quadlist, i);
			bt_insert(me->bigquadlist, q, FALSE, compare_quads);
		}
		bl_remove_all(me->quadlist);
	}

	if (Nloosen) {
		int R;
		for (R=Nreuses+1; R<=Nloosen; R++) {
			il* trylist;
			int nthispass;

			printf("Loosening reuse maximum to %i...\n", R);
			printf("Trying %i healpixes.\n", il_size(me->retryhps));
			if (!il_size(me->retryhps))
				break;

			trylist = me->retryhps;
			me->retryhps = il_new(1024);
			nthispass = build_quads(me, il_size(trylist), trylist, R);
			printf("Made %i quads (out of %i healpixes) this pass.\n", nthispass, il_size(trylist));
			for (i=0; i<bl_size(me->quadlist); i++) {
				void* q = bl_access(me->quadlist, i);
				bt_insert(me->bigquadlist, q, FALSE, compare_quads);
			}
			bl_remove_all(me->quadlist);
		}
	}
	if (me->retryhps)
		il_free(me->retryhps);

	kdtree_free_query(me->res);
	me->res = NULL;
	me->inds = NULL;
	me->stars = NULL;

	printf("Writing quads...\n");

	// add the quads from the big-quadlist
	nquads = bt_size(me->bigquadlist);
	for (i=0; i<nquads; i++) {
		unsigned int* q = bt_access(me->bigquadlist, i);
		quad_write(me->codes, me->quads, q, me->starkd, me->dimquads, me->dimcodes);
	}
	// add the quads that were made during the final round.
	for (i=0; i<bl_size(me->quadlist); i++) {
		unsigned int* q = bl_access(me->quadlist, i);
		quad_write(me->codes, me->quads, q, me->starkd, me->dimquads, me->dimcodes);
	}

	// fix output file headers.
	if (quadfile_fix_header(me->quads) ||
		quadfile_close(me->quads)) {
		ERROR("Couldn't write quad output file");
		return -1;
	}
	if (codefile_fix_header(me->codes) ||
		codefile_close(me->codes)) {
		ERROR("Couldn't write code output file");
		return -1;
	}
	
	bl_free(me->quadlist);
	bt_free(me->bigquadlist);
	startree_close(me->starkd);

	toc();
	printf("Done.\n");
	return 0;
}

