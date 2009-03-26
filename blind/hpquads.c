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
#include "rdlist.h"
#include "histogram.h"
#include "starkd.h"
#include "pnpoly.h"
#include "boilerplate.h"
#include "errors.h"

#define OPTIONS "hi:u:l:n:I:r:x:y:F:RHL:bc:q:d:E"

static void print_help(char* progname)
{
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
	       "      -i <input-filename>    (star kdtree (skdt.fits) input file)\n"
		   "      -c <codes-output-filename>    (codes file (code.fits) output file)\n"
           "      -q <quads-output-filename>    (quads file (quad.fits) output file)\n"
		   "     [-b]            try to make any quads outside the bounds of this big healpix.\n"
	       "     [-n <nside>]    healpix nside (default 501)\n"
	       "     [-u <scale>]    upper bound of quad scale (arcmin)\n"
	       "     [-l <scale>]    lower bound of quad scale (arcmin)\n"
		   "     [-d <dimquads>] number of stars in a \"quad\".\n"
		   "     [-x <x-passes>] number of passes in the x direction\n"
		   "     [-y <y-passes>] number of passes in the y direction\n"
		   "     [-r <reuse-times>] number of times a star can be used.\n"
		   "     [-L <max-reuses>] make extra passes through the healpixes, increasing the \"-r\" reuse\n"
		   "                     limit each time, up to \"max-reuses\".\n"
		   "     [-R]            make a second pass through healpixes in which quads couldn't be made,\n"
		   "                     removing the \"-r\" restriction on the number of times a star can be used.\n"
		   "     [-I <unique-id>] set the unique ID of this index\n\n"
		   "     [-F <failed-rdls-file>] write the centers of the healpixes in which quads can't be made.\n"
		   "     [-H]: print histograms.\n"
		   "     [-E]: scan through the catalog, checking which healpixes are occupied.\n"
		   "\nReads skdt, writes {code, quad}.\n\n"
	       , progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

static quadfile* quads;
static codefile* codes;
static startree_t* starkd;

// bounds of quad scale (in radians^2)
static double quad_scale_upper2;
static double quad_scale_lower2;

// bounds of quad scale, in distance between AB on the sphere.
static double quad_dist2_upper;
static double quad_dist2_lower;

struct quad {
	unsigned int star[DQMAX];
};
typedef struct quad quad;

static int Nquads;
static quad* quadlist;
static bt* bigquadlist;

static int ndupquads = 0;
static int nbadscale = 0;
static int nbadcenter = 0;
static int nabok = 0;

static unsigned char* nuses;

static bool hists = FALSE;

static void* mymalloc(unsigned int n, int linenum) {
	void* rtn = malloc(n);
	if (!rtn) {
		fprintf(stderr, "mymalloc failed: line number %i: %u bytes.\n", linenum, n);
		assert(0);
		exit(-1);
	}
	return rtn;
}

static int compare_ints(const void* v1, const void* v2) {
	int i1 = *(int*)v1;
	int i2 = *(int*)v2;
	if (i1 < i2) return -1;
	if (i1 > i2) return 1;
	return 0;
}

static int compare_quads(const void* v1, const void* v2) {
	const quad* q1 = v1;
	const quad* q2 = v2;
	int i;
	// Hmm... I thought about having a static global "dimquads" here, but
	// instead just ensured that are quad is always initialized to zero so that
	// "star" values between dimquads and DQMAX are always equal.
	for (i=0; i<DQMAX; i++) {
		if (q1->star[i] > q2->star[i])
			return 1;
		if (q1->star[i] < q2->star[i])
			return -1;
	}
	return 0;
}

static bool firstpass;

static bool add_quad(quad* q) {
	if (!firstpass) {
		bool dup = bt_contains(bigquadlist, q, compare_quads);
		if (dup) {
			ndupquads++;
			return FALSE;
		}
	}
	quadlist[Nquads++] = *q;
	return TRUE;
}

static void compute_code(quad* q, double* code, int dimquads) {
    int i;
	double starxyz[3 * DQMAX];

	for (i=0; i<dimquads; i++) {
		if (startree_get(starkd, q->star[i], starxyz + 3*i)) {
			fprintf(stderr, "Failed to get stars belonging to a quad.\n");
			exit(-1);
		}
	}

    codefile_compute_star_code(starxyz, code, dimquads);
}

static Inline void drop_quad(quad* q, int dimquads) {
	int i;
	for (i=0; i<dimquads; i++)
		nuses[q->star[i]]++;
}

static void write_quad(codefile* codes, quadfile* quads,
					   quad* q, int dimquads, int dimcodes) {
	double code[DCMAX];
	double sum;
	int i;

	compute_code(q, code, dimquads);

	// here we add the invariant that (cx + dx + ...) / (dimquads-2) <= 1/2
	sum = 0.0;
	for (i=0; i<(dimquads-2); i++)
		sum += code[2*i];
	sum /= (dimquads-2);
	if (sum > 0.5) {
		// swap the labels of A,B.
		int tmp = q->star[0];
		q->star[0] = q->star[1];
		q->star[1] = tmp;

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
		tmp = q->star[i+2];
		q->star[i+2] = q->star[j+2];
		q->star[j+2] = tmp;
		// swap the code values.
		dtmp = code[2*i];
		code[2*i] = code[2*j];
		code[2*j] = dtmp;
		dtmp = code[2*i+1];
		code[2*i+1] = code[2*j+1];
		code[2*j+1] = dtmp;
	}

	codefile_write_code(codes, code);
	quadfile_write_quad(quads, q->star);
}

struct potential_quad {
	double midAB[3];
	double Ax, Ay;
	double costheta, sintheta;
	int iA, iB;
	int staridA, staridB;
	int* inbox;
	int ninbox;
	bool scale_ok;
};
typedef struct potential_quad pquad;

// is the AB distance right?
// is the midpoint of AB inside the healpix?
static void
check_scale_and_midpoint(pquad* pq, double* stars, int* starids, int Nstars,
						 double* origin, double* vx, double* vy,
						 double* boxx, double* boxy) {
	double *sA, *sB;
	double Bx, By;
	double invscale;
	double ABx, ABy;
	double px, py;
	double s2;
	int d;
	double avgAB[3];
	double invlen;
	bool ok;

	sA = stars + pq->iA * 3;
	sB = stars + pq->iB * 3;

	// s2: squared AB dist
	s2 = 0.0;
	for (d=0; d<3; d++)
		s2 += (sA[d] - sB[d])*(sA[d] - sB[d]);
	if ((s2 > quad_dist2_upper) ||
		(s2 < quad_dist2_lower)) {
		pq->scale_ok = 0;
		nbadscale++;
		return;
	}

	// avgAB: mean of A,B.  (note, it's NOT on the sphere, and that's okay
	// because we're projecting it into the ~tangent plane of the target
	// healpix)
	for (d=0; d<3; d++)
		avgAB[d] = 0.5 * (sA[d] + sB[d]);

	px = 
		(avgAB[0] - origin[0]) * vx[0] +
		(avgAB[1] - origin[1]) * vx[1] +
		(avgAB[2] - origin[2]) * vx[2];
	py =
		(avgAB[0] - origin[0]) * vy[0] +
		(avgAB[1] - origin[1]) * vy[1] +
		(avgAB[2] - origin[2]) * vy[2];

	if (!point_in_poly(boxx, boxy, 4, px, py)) {
		pq->scale_ok = 0;
		nbadcenter++;
		return;
	}

	pq->scale_ok = 1;
	pq->staridA = starids[pq->iA];
	pq->staridB = starids[pq->iB];

	invlen = 1.0 / sqrt(avgAB[0]*avgAB[0] + avgAB[1]*avgAB[1] + avgAB[2]*avgAB[2]);
	pq->midAB[0] = avgAB[0] * invlen;
	pq->midAB[1] = avgAB[1] * invlen;
	pq->midAB[2] = avgAB[2] * invlen;
	ok = star_coords(sA, pq->midAB, &pq->Ax, &pq->Ay);
	assert(ok);
	ok = star_coords(sB, pq->midAB, &Bx, &By);
	assert(ok);
	ABx = Bx - pq->Ax;
	ABy = By - pq->Ay;
	invscale = 1.0 / (ABx*ABx + ABy*ABy);
	pq->costheta = (ABy + ABx) * invscale;
	pq->sintheta = (ABy - ABx) * invscale;

	nabok++;
}

static int
check_inbox(pquad* pq, int* inds, int ninds, double* stars, bool circle) {
	int i, ind;
	double* starpos;
	double Dx, Dy;
	double ADx, ADy;
	double x, y;
	int destind = 0;
	bool ok;
	for (i=0; i<ninds; i++) {
		ind = inds[i];
		starpos = stars + ind*3;
		ok = star_coords(starpos, pq->midAB, &Dx, &Dy);
		if (!ok)
			continue;
		ADx = Dx - pq->Ax;
		ADy = Dy - pq->Ay;
		x =  ADx * pq->costheta + ADy * pq->sintheta;
		y = -ADx * pq->sintheta + ADy * pq->costheta;
		if (circle) {
			// make sure it's in the circle centered at (0.5, 0.5)...
			// (x-1/2)^2 + (y-1/2)^2   <=   r^2
			// x^2-x+1/4 + y^2-y+1/4   <=   (1/sqrt(2))^2
			// x^2-x + y^2-y + 1/2     <=   1/2
			// x^2-x + y^2-y           <=   0
			double r = (x*x - x) + (y*y - y);
			if (r > 0.0)
				continue;
		} else {
			// make sure it's in the box...
			if ((x > 1.0) || (x < 0.0) ||
				(y > 1.0) || (y < 0.0)) {
				continue;
			}
		}
		inds[destind] = ind;
		destind++;
	}
	return destind;
}

static int add_interior_stars(int ninbox, int* inbox, quad* q, int* starinds,
							  int starnum, int dimquads, int beginning) {
	int i;
	for (i=beginning; i<ninbox; i++) {
		int iC = inbox[i];
		q->star[starnum] = starinds[iC];
		// Did we just add the last star?
		if (starnum == dimquads-1) {
			if (add_quad(q))
				return 1;
		} else {
			// Recurse.
			if (add_interior_stars(ninbox, inbox, q, starinds, starnum+1,
								   dimquads, i+1))
				return 1;
		}
	}
	return 0;
}

static int Ncq = 0;
static pquad* cq_pquads = NULL;
static int* cq_inbox = NULL;

static int create_quad(double* stars, int* starinds, int Nstars,
					   bool circle,
					   double* origin, double* vx, double* vy,
					   double* boxx, double* boxy,
					   bool count_uses, int dimquads) {
	int iA=0, iB, iC, iD, newpoint;
	int rtn = 0;
	int ninbox;
	int i, j;
	int* inbox;
	pquad* pquads;
	int iAalloc;
	quad q;

	// ensure the arrays are large enough...
	if (Nstars > Ncq) {
		// (free and malloc rather than realloc because we don't care about
		//  the previous contents)
		free(cq_inbox);
		free(cq_pquads);
		Ncq = Nstars;
		cq_inbox =  malloc(Nstars * sizeof(int));
		cq_pquads = malloc(Nstars * Nstars * sizeof(pquad));
		if (!cq_inbox || !cq_pquads) {
			fprintf(stderr, "hpquads: failed to malloc cq_inbox or cq_pquads.  Nstars=%i.\n", Nstars);
			exit(-1);
		}
	}
	inbox = cq_inbox;
	pquads = cq_pquads;

	/*
	  Each time through the "for" loop below, we consider a new
	  star ("newpoint").  First, we try building all quads that
	  have the new star on the diagonal (star B).  Then, we try
	  building all quads that have the star not on the diagonal
	  (star D).

	  Note that we keep the invariants iA < iB and iC < iD.
	*/

	memset(&q, 0, sizeof(quad));

	for (newpoint=0; newpoint<Nstars; newpoint++) {
		pquad* pq;
		// quads with the new star on the diagonal:
		iB = newpoint;
		for (iA = 0; iA < newpoint; iA++) {
			pq = pquads + iA*Nstars + iB;
			pq->inbox = NULL;
			pq->ninbox = 0;
			pq->iA = iA;
			pq->iB = iB;
			check_scale_and_midpoint(pq, stars, starinds, Nstars,
									 origin, vx, vy, boxx, boxy);
			if (!pq->scale_ok)
				continue;

			ninbox = 0;
			for (iC = 0; iC < newpoint; iC++) {
				if ((iC == iA) || (iC == iB))
					continue;
				inbox[ninbox] = iC;
				ninbox++;
			}
			ninbox = check_inbox(pq, inbox, ninbox, stars, circle);

			q.star[0] = pq->staridA;
			q.star[1] = pq->staridB;

			if (add_interior_stars(ninbox, inbox, &q, starinds, 2, dimquads, 0)) {
				if (count_uses)
					drop_quad(&q, dimquads);
				rtn = 1;
				goto theend;
			}

			pq->inbox = malloc(Nstars * sizeof(int));
			if (!pq->inbox) {
				fprintf(stderr, "hpquads: failed to malloc pq->inbox.\n");
				exit(-1);
			}
			pq->ninbox = ninbox;
			memcpy(pq->inbox, inbox, ninbox * sizeof(int));
		}
		iAalloc = iA;

		// quads with the new star not on the diagonal:
		iD = newpoint;
		for (iA = 0; iA < newpoint; iA++) {
			for (iB = iA + 1; iB < newpoint; iB++) {
				pq = pquads + iA*Nstars + iB;
				if (!pq->scale_ok)
					continue;
				inbox[0] = iD;
				if (!check_inbox(pq, inbox, 1, stars, circle))
					continue;
				pq->inbox[pq->ninbox] = iD;
				pq->ninbox++;
				ninbox = pq->ninbox;

				q.star[0] = pq->staridA;
				q.star[1] = pq->staridB;

				if (add_interior_stars(ninbox, pq->inbox, &q, starinds,
									   2, dimquads, 0)) {
					if (count_uses)
						drop_quad(&q, dimquads);
					rtn = 1;
					iA = iAalloc;
					goto theend;
				}
			}
		}
	}
 theend:
	for (i=0; i<imin(Nstars, newpoint+1); i++) {
		int lim = (i == newpoint) ? iA : i;
		for (j=0; j<lim; j++) {
			pquad* pq = pquads + j*Nstars + i;
			free(pq->inbox);
		}
	}
	return rtn;
}

static int* perm = NULL;
static int* inds = NULL;
static double* stars = NULL;

static bool find_stars_and_vectors(int hp, int Nside, double radius2,
								   double dxfrac, double dyfrac,
								   int* p_nostars, int* p_yesstars,
								   int* p_nounused, int* p_nstarstotal,
								   int* p_ncounted,
								   int* p_N,
								   double* centre, double* vx, double* vy,
								   double* xpts, double* ypts,
								   bool* p_failed_nostars,
								   int R,
								   int dimquads) {
	static int Nhighwater = 0;
	double origin[3];
	double dx[3];
	double dy[3];
	double dxy[3];
	double normal[3];
	int d;
	kdtree_qres_t* res;
	int j, N;
	int destind;

	healpix_to_xyzarr(hp, Nside, 0.0, 0.0, origin);
	healpix_to_xyzarr(hp, Nside, 1.0, 0.0, dx);
	healpix_to_xyzarr(hp, Nside, 0.0, 1.0, dy);
	healpix_to_xyzarr(hp, Nside, 1.0, 1.0, dxy);
	for (d=0; d<3; d++) {
		dx[d] -= origin[d];
		dy[d] -= origin[d];
		dxy[d] -= origin[d];
		centre[d] = origin[d] + dxfrac*dx[d] + dyfrac*dy[d];
	}

	res = kdtree_rangesearch_nosort(starkd->tree, centre, radius2);

	// here we could check whether stars are in the box defined by the
	// healpix boundaries plus quad scale, rather than just the circle
	// containing that box.

	N = res->nres;
	if (N < dimquads) {
		kdtree_free_query(res);
		if (p_nostars)
			(*p_nostars)++;
		if (p_N) *p_N = N;
		if (p_failed_nostars)
			*p_failed_nostars = TRUE;
		return FALSE;
	}
	if (p_yesstars)
		(*p_yesstars)++;

	// remove stars that have been used up.
	destind = 0;
	for (j=0; j<N; j++) {
		if (nuses[res->inds[j]] >= R)
			continue;
		res->inds[destind] = res->inds[j];
		for (d=0; d<3; d++)
			res->results.d[destind*3+d] = res->results.d[j*3+d];
		destind++;
	}
	N = destind;
	if (N < dimquads) {
		kdtree_free_query(res);
		if (p_nounused)
			(*p_nounused)++;
		if (p_N) *p_N = N;
		return FALSE;
	}

	if (p_nstarstotal)
		(*p_nstarstotal) += N;
	if (p_ncounted)
		(*p_ncounted)++;

	// sort the stars in increasing order of index - assume
	// that this corresponds to decreasing order of brightness.

	// ensure the arrays are big enough...
	if (N > Nhighwater) {
		free(perm);
		free(inds);
		free(stars);
		perm  = mymalloc(N * sizeof(int), __LINE__);
		inds  = mymalloc(N * sizeof(int), __LINE__);
		stars = mymalloc(N * 3 * sizeof(double), __LINE__);
		Nhighwater = N;
	}
	// find permutation that sorts by index...
	permutation_init(perm, N);
	permuted_sort(res->inds, sizeof(int), compare_ints, perm, N);
	// apply the permutation...
	permutation_apply(perm, N, res->inds, inds, sizeof(int));
	permutation_apply(perm, N, res->results.d, stars, 3 * sizeof(double));
	kdtree_free_query(res);

	// compute the projection vectors
	cross_product(dx, dy, normal);
	normalize_3(normal);
	cross_product(normal, dx, vy);
	normalize_3(vy);
	cross_product(vy, normal, vx);
	// (this should already be normalized!)
	normalize_3(vx);

	xpts[0] = 0.0;
	ypts[0] = 0.0;
	xpts[1] = dot_product_3(vx, dx);
	ypts[1] = dot_product_3(vy, dx);
	xpts[2] = dot_product_3(vx, dxy);
	ypts[2] = dot_product_3(vy, dxy);
	xpts[3] = dot_product_3(vx, dy);
	ypts[3] = dot_product_3(vy, dy);

	if (p_N) *p_N = N;
	return TRUE;
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

int main(int argc, char** argv) {
	int argchar;
	char *quadfname = NULL;
	char *codefname = NULL;
	char *skdtfname = NULL;
	int64_t HEALPIXES;
	int Nside = 501;
	int i;
	char* failedrdlsfn = NULL;
	rdlist_t* failedrdls = NULL;
	int xpass, ypass;
	int id = 0;
	int xpasses = 1;
	int ypasses = 1;
	bool circle = TRUE;
	int Nreuse = 3;
	double radius2;
	int lastgrass = 0;
	ll* hptotry;
	int nquads;
	double rads;
	double hprad;
	double quadscale;
	bool noreuse_pass = FALSE;
	il* noreuse_hps = NULL;

	dl* nostars_radec = NULL;
	dl* noreuse_radec = NULL;
	dl* noquads_radec = NULL;

	int loosenmax = 0;
	il** loosenhps = NULL;

	double dxfrac, dyfrac;
	double centre[3];
	double vx[3], vy[3];
	double boxx[4], boxy[4];
	int hp, hpnside;

	qfits_header* qhdr;
	qfits_header* chdr;

	bool boundary = FALSE;
	bool scanoccupied = FALSE;

	int dimquads = 4;
	int dimcodes;
	
	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'E':
			scanoccupied = TRUE;
			break;
		case 'd':
			dimquads = atoi(optarg);
			break;
		case 'b':
			boundary = TRUE;
			break;
		case 'L':
			loosenmax = atoi(optarg);
			break;
		case 'R':
			noreuse_pass = TRUE;
			break;
		case 'F':
			failedrdlsfn = optarg;
			break;
		case 'r':
			Nreuse = atoi(optarg);
			break;
		case 'x':
			xpasses = atoi(optarg);
			break;
		case 'y':
			ypasses = atoi(optarg);
			break;
		case 'I':
			id = atoi(optarg);
			break;
		case 'n':
			Nside = atoi(optarg);
			break;
		case 'h':
			print_help(argv[0]);
			exit(0);
		case 'i':
            skdtfname = optarg;
			break;
		case 'c':
            codefname = optarg;
            break;
        case 'q':
            quadfname = optarg;
            break;
		case 'u':
			rads = arcmin2rad(atof(optarg));
			quad_scale_upper2 = square(rads);
			quad_dist2_upper = square(tan(rads));
			break;
		case 'l':
			rads = arcmin2rad(atof(optarg));
			quad_scale_lower2 = square(rads);
			quad_dist2_lower = square(tan(rads));
			break;
		case 'H':
			hists = TRUE;
			break;
		default:
			return -1;
		}

	if (!skdtfname || !codefname || !quadfname) {
		fprintf(stderr, "Specify in & out filenames, bonehead!\n");
		print_help(argv[0]);
		exit( -1);
	}

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

	if (!id) {
		fprintf(stderr, "Warning: you should set the unique-id for this index (-i).\n");
	}

	if (dimquads > DQMAX) {
		fprintf(stderr, "Quad dimension %i exceeds compiled-in max %i.\n", dimquads, DQMAX);
		exit(-1);
	}
	dimcodes = dimquad2dimcode(dimquads);

	if (failedrdlsfn) {
		failedrdls = rdlist_open_for_writing(failedrdlsfn);
		if (!failedrdls) {
			fprintf(stderr, "Failed to open file %s to write failed-quads RDLS.\n", failedrdlsfn);
			exit(-1);
		}
		if (rdlist_write_primary_header(failedrdls)) {
			fprintf(stderr, "Failed to write header of failed RDLS file.\n");
		}
	}

	if (loosenmax)
		loosenhps = malloc(xpasses * ypasses * sizeof(il*));

	HEALPIXES = 12L * (int64_t)Nside * (int64_t)Nside;
	printf("Nside=%i.  Nside^2=%lli.  Number of healpixes=%lli.  Healpix side length ~ %g arcmin.\n",
		   Nside, (long long int)((int64_t)Nside*(int64_t)Nside),
		   (long long int)HEALPIXES, healpix_side_length_arcmin(Nside));

	tic();

	printf("Reading star kdtree %s ...\n", skdtfname);
	starkd = startree_open(skdtfname);
	if (!starkd) {
		fprintf(stderr, "Failed to open star kdtree %s\n", skdtfname);
		exit( -1);
	}
	printf("Star tree contains %i objects.\n", startree_N(starkd));

	printf("Will write to quad file %s and code file %s\n", quadfname, codefname);

    quads = quadfile_open_for_writing(quadfname);
	if (!quads) {
		fprintf(stderr, "Couldn't open file %s to write quads.\n", quadfname);
		exit(-1);
	}
    codes = codefile_open_for_writing(codefname);
	if (!codes) {
		fprintf(stderr, "Couldn't open file %s to write codes.\n", quadfname);
		exit(-1);
	}

	quads->dimquads = dimquads;
	codes->dimcodes = dimcodes;

	if (id) {
		quads->indexid = id;
		codes->indexid = id;
	}

	// get the "HEALPIX" header from the skdt and put it in the code and quad headers.
	hp = qfits_header_getint(startree_header(starkd), "HEALPIX", -1);
	if (hp == -1) {
		fprintf(stderr, "Warning: skdt does not contain \"HEALPIX\" header.  Code and quad files will not contain this header either.\n");
	}
	quads->healpix = hp;
	codes->healpix = hp;
    // likewise "HPNSIDE"
	hpnside = qfits_header_getint(startree_header(starkd), "HPNSIDE", 1);
	quads->hpnside = hpnside;
	codes->hpnside = hpnside;

    if (hpnside && Nside % hpnside) {
        fprintf(stderr, "Error: Nside (-n) must be a multiple of the star kdtree healpixelisation: %i\n", hpnside);
        exit(-1);
    }

	qhdr = quadfile_get_header(quads);
	chdr = codefile_get_header(codes);

	add_headers(qhdr, argv, argc, startree_header(starkd), circle, xpasses*ypasses);
	add_headers(chdr, argv, argc, startree_header(starkd), circle, xpasses*ypasses);

    if (quadfile_write_header(quads)) {
        fprintf(stderr, "Couldn't write headers to quads file %s\n", quadfname);
        exit(-1);
    }
    if (codefile_write_header(codes)) {
        fprintf(stderr, "Couldn't write headers to code file %s\n", codefname);
        exit(-1);
    }

    codes->numstars = startree_N(starkd);
    codes->index_scale_upper = sqrt(quad_scale_upper2);
    codes->index_scale_lower = sqrt(quad_scale_lower2);

    quads->numstars = startree_N(starkd);
    quads->index_scale_upper = sqrt(quad_scale_upper2);
    quads->index_scale_lower = sqrt(quad_scale_lower2);

	bigquadlist = bt_new(sizeof(quad), 256);

	if (Nreuse > 255) {
		fprintf(stderr, "Error, reuse (-r) must be less than 256.\n");
		exit(-1);
	}
	nuses = mymalloc(startree_N(starkd) * sizeof(unsigned char), __LINE__);
	for (i=0; i<startree_N(starkd); i++)
		nuses[i] = 0;

	hprad = sqrt(0.5 * arcsec2distsq(healpix_side_length_arcmin(Nside) * 60.0));
	quadscale = 0.5 * sqrt(quad_dist2_upper);
	// 1.01 for a bit of safety.  we'll look at a few extra stars.
	radius2 = square(1.01 * (hprad + quadscale));

	printf("Healpix radius %g arcsec, quad scale %g arcsec, total %g arcsec\n",
		   distsq2arcsec(hprad*hprad),
		   distsq2arcsec(quadscale*quadscale),
		   distsq2arcsec(radius2));


	hptotry = ll_new(256);

	if (scanoccupied) {
		int i, N;
		N = startree_N(starkd);
		printf("Scanning %i input stars...\n", N);
		for (i=0; i<N; i++) {
			double xyz[3];
			int64_t j;
			if (startree_get(starkd, i, xyz)) {
				ERROR("Failed to get star %i", i);
				exit(-1);
			}
			j = xyzarrtohealpixl(xyz, Nside);
			ll_insert_unique_ascending(hptotry, j);
		}
		printf("Will check %i healpixes.\n", ll_size(hptotry));
	} else {
		if (hp == -1) {
			int64_t j;
			// Try all healpixes.
			for (j=0; j<HEALPIXES; j++)
				ll_append(hptotry, j);
		} else {
			// The star kdtree may itself be healpixed
			int starhp, starx, stary;
			// In that case, the healpixes we are interested in form a rectangle
			// within a big healpix.  These are the coords (in [0, Nside)) of
			// that rectangle.
			int x0, x1, y0, y1;
			int x, y;
			int nhp;

			healpix_decompose_xy(hp, &starhp, &starx, &stary, hpnside);
			x0 =  starx    * (Nside / hpnside);
			x1 = (starx+1) * (Nside / hpnside);
			y0 =  stary    * (Nside / hpnside);
			y1 = (stary+1) * (Nside / hpnside);

			nhp = 0;
			for (y=y0; y<y1; y++) {
				for (x=x0; x<x1; x++) {
					int64_t j = healpix_compose_xyl(starhp, x, y, Nside);
					ll_append(hptotry, j);
				}
			}
			assert(ll_size(hptotry) == (Nside/hpnside) * (Nside/hpnside));
		}
	}

	quadlist = malloc(ll_size(hptotry) * sizeof(quad));

	if (noreuse_pass)
		noreuse_hps = il_new(1024);

	if (failedrdls) {
		nostars_radec = dl_new(1024);
		noreuse_radec = dl_new(1024);
		noquads_radec = dl_new(1024);
	}

	firstpass = TRUE;
	Nquads = 0;

	for (xpass=0; xpass<xpasses; xpass++) {
		for (ypass=0; ypass<ypasses; ypass++) {
			int nthispass;
			int nnostars;
			int nyesstars;
			int nnounused;
			int nstarstotal = 0;
			int ncounted = 0;
			il* loosen = NULL;

			histogram* histnstars = NULL;
			histogram* histnstars_failed = NULL;

			if (hists) {
				histnstars = histogram_new_nbins(0.0, 100.0, 100);
				histnstars_failed = histogram_new_nbins(0.0, 100.0, 100);
			}

			if (loosenmax) {
				loosen = loosenhps[xpass * ypasses + ypass] = il_new(1024);
			}

			dxfrac = xpass / (double)xpasses;
			dyfrac = ypass / (double)ypasses;

			printf("Pass %i of %i.\n", xpass * ypasses + ypass + 1, xpasses * ypasses);
			nthispass = 0;
			nnostars = 0;
			nyesstars = 0;
			nnounused = 0;
			lastgrass = 0;
			nbadscale = 0;
			nbadcenter = 0;
			nabok = 0;
			ndupquads = 0;

			printf("Trying %i healpixes.\n", ll_size(hptotry));

			for (i=0; i<ll_size(hptotry); i++) {
				double radec[2];
				int N;
				bool ok;
				bool failed_nostars;

				hp = ll_get(hptotry, i);

				if ((i * 80 / ll_size(hptotry)) != lastgrass) {
					printf(".");
					fflush(stdout);
					lastgrass = i * 80 / ll_size(hptotry);
				}

				failed_nostars = FALSE;
				ok = find_stars_and_vectors(hp, Nside, radius2,
											dxfrac, dyfrac,
											&nnostars, &nyesstars,
											&nnounused, &nstarstotal,
											&ncounted,
											&N, centre, vx, vy,
											boxx, boxy,
											&failed_nostars,
											Nreuse, dimquads);

				if (failedrdls) {
					xyz2radec(centre[0], centre[1], centre[2], radec, radec+1);
					radec[0] = rad2deg(radec[0]);
					radec[1] = rad2deg(radec[1]);
				}

				if (!ok) {
					// Did we fail because there were no stars?
					if (failed_nostars) {
						if (failedrdls) {
							dl_append(nostars_radec, radec[0]);
							dl_append(nostars_radec, radec[1]);
						}
					} else {
						if (noreuse_pass)
							il_append(noreuse_hps, hp);
						if (loosen)
							il_append(loosen, hp);
						if (failedrdls) {
							dl_append(noreuse_radec, radec[0]);
							dl_append(noreuse_radec, radec[1]);
						}
					}
					if (histnstars_failed)
						histogram_add(histnstars_failed, (double)N);
					continue;
				}

				if (create_quad(stars, inds, N, circle,
								centre, vx, vy, boxx, boxy, TRUE, dimquads)) {
					if (histnstars)
						histogram_add(histnstars, (double)N);
					nthispass++;
				} else {
					if (noreuse_pass)
						il_append(noreuse_hps, hp);
					if (loosen)
						il_append(loosen, hp);
					if (failedrdls) {
						dl_append(noquads_radec, radec[0]);
						dl_append(noquads_radec, radec[1]);
					}
					if (histnstars_failed)
						histogram_add(histnstars_failed, (double)N);
				}
			}
			printf("\n");

			if (hists) {
				printf("Number of stars per healpix histogram:\n");
				printf("hist_nstars=");
				histogram_print_matlab(histnstars, stdout);
				printf("\n");
				printf("hist_nstars_bins=");
				histogram_print_matlab_bin_centers(histnstars, stdout);
				printf("\n");
				printf("Number of stars per healpix, for failed healpixes:\n");
				printf("hist_nstars_failed=");
				histogram_print_matlab(histnstars_failed, stdout);
				printf("\n");
				printf("hist_nstars_failed_bins=");
				histogram_print_matlab_bin_centers(histnstars_failed, stdout);
				printf("\n");

				histogram_free(histnstars);
				histogram_free(histnstars_failed);
			}

			printf("Each non-empty healpix had on average %g stars.\n",
				   nstarstotal / (double)ncounted);

			printf("Made %i quads (out of %i healpixes) this pass.\n",
				   nthispass, ll_size(hptotry));
			printf("  %i healpixes had no stars.\n", nnostars);
			printf("  %i healpixes had only stars that had been overused.\n", nnounused);
			printf("  %i healpixes had some stars.\n", nyesstars);
			printf("  %i AB pairs had bad scale.\n", nbadscale);
			printf("  %i AB pairs had bad center.\n", nbadcenter);
			printf("  %i AB pairs were ok.\n", nabok);
			printf("  %i quads were duplicates.\n", ndupquads);

			{
				char key[64];
				sprintf(key, "PASS%i", xpass * ypasses + ypass + 1);
				fits_header_mod_int(chdr, key, nthispass, "quads created in this pass");
				fits_header_mod_int(qhdr, key, nthispass, "quads created in this pass");
			}

			// HACK -
			// sort the quads in "quadlist", then insert them into
			// "bigquadlist" ?

			printf("Made %i quads so far.\n", bt_size(bigquadlist) + Nquads);

			if (failedrdls) {
				//int j;
				dl* lists[] = { nostars_radec, noreuse_radec, noquads_radec };
				int l;

				for (l=0; l<3; l++) {
					dl* list = lists[l];
                    rd_t rd;
                    /*
                     if (rdlist_write_new_field(failedrdls)) {
                     fprintf(stderr, "Failed to start a new field in failed RDLS file.\n");
                     exit(-1);
                     }
                     */
					if (rdlist_write_header(failedrdls)) {
						fprintf(stderr, "Failed to write a field in failed RDLS file.\n");
						exit(-1);
					}

                    rd_from_dl(&rd, list);
                    rdlist_write_field(failedrdls, &rd);
                    rd_free_data(&rd);
                    /*
                     for (j=0; j<dl_size(list); j+=2) {
                     double radec[2];
                     radec[0] = dl_get(list, j);
                     radec[1] = dl_get(list, j+1);
                     if (rdlist_write_entries(failedrdls, radec, 1)) {
                     fprintf(stderr, "Failed to write failed-RDLS entries.\n");
                     exit(-1);
                     }
                     }
                     */
					if (rdlist_fix_header(failedrdls)) {
						fprintf(stderr, "Failed to fix a field in failed RDLS file.\n");
						exit(-1);
					}
                    rdlist_next_field(failedrdls);
				}
			}

			if (noreuse_pass) {
				int i;
				int nfailed1 = 0;
				int nfailed2 = 0;
				int nmade = 0;
				lastgrass = -1;
				if (failedrdls) {
					if (rdlist_write_header(failedrdls)) {
						fprintf(stderr, "Failed to start a new field in failed RDLS file.\n");
						exit(-1);
					}
				}
				if (il_size(noreuse_hps)) {
					printf("Making a pass with no limit on the number of times a star can be used.\n");
					printf("Trying %i healpixes.\n", il_size(noreuse_hps));
					for (i=0; i<il_size(noreuse_hps); i++) {
						int N;
						int hp = il_get(noreuse_hps, i);

						if ((i * 80 / il_size(noreuse_hps)) != lastgrass) {
							printf(".");
							fflush(stdout);
							lastgrass = i * 80 / il_size(noreuse_hps);
						}

						if (!find_stars_and_vectors(hp, Nside, radius2,
													dxfrac, dyfrac,
													NULL, NULL, NULL, NULL, NULL,
													&N, centre, vx, vy,
													boxx, boxy,
													NULL, INT_MAX, dimquads)) {
							nfailed1++;
							goto failedhp2;
						}
						if (!create_quad(stars, inds, N, circle,
										 centre, vx, vy, boxx, boxy, FALSE, dimquads)) {
							nfailed2++;
							goto failedhp2;
						}
						nmade++;
						continue;
					failedhp2:
						if (failedrdls) {
                            double ra, dec;
                            rd_t rd;
                            xyzarr2radecdeg(centre, &ra, &dec);
                            rd.N = 1;
                            rd.ra  = &ra;
                            rd.dec = &dec;
							if (rdlist_write_field(failedrdls, &rd)) {
								fprintf(stderr, "Failed to write failed-RDLS entries.\n");
								exit(-1);
							}
						}
					}

					printf("\n");
					printf("Tried %i healpixes.\n", il_size(noreuse_hps));
					printf("Failed at point 1: %i.\n", nfailed1);
					printf("Failed at point 2: %i.\n", nfailed2);
					printf("Made: %i\n", nmade);
					il_remove_all(noreuse_hps);
				}
				if (failedrdls) {
					if (rdlist_fix_header(failedrdls)) {
						fprintf(stderr, "Failed to fix a field in failed RDLS file.\n");
						exit(-1);
					}
				}
			}

			printf("Merging quads...\n");
			for (i=0; i<Nquads; i++) {
				quad* q = quadlist + i;
				bt_insert(bigquadlist, q, FALSE, compare_quads);
			}
			Nquads = 0;

			firstpass = FALSE;

		}
	}
	ll_free(hptotry);


	if (loosenmax) {
		int mx;
		// toggle the direction in which we make the passes...
		int dir = 1;
		for (mx=Nreuse+1; mx<=loosenmax; mx++) {
			int alldone = 1;
			dir = !dir;
			printf("Loosening reuse maximum to %i...\n", mx);
			for (xpass=(dir ? 0 : xpasses-1); (dir ? xpass<xpasses : xpass>=0); (dir ? xpass++ : xpass--)) {
				for (ypass=(dir ? 0 : ypasses-1); (dir ? ypass<ypasses : ypass>=0); (dir ? ypass++ : ypass--)) {
					il* loosen;
					il* newlist;
					int nmade = 0;

					loosen = loosenhps[xpass * ypasses + ypass];
					if (!loosen)
						continue;

					alldone = 0;
					dxfrac = xpass / (double)xpasses;
					dyfrac = ypass / (double)ypasses;

					printf("Pass %i of %i.\n", xpass * ypasses + ypass + 1, xpasses * ypasses);
					printf("Trying %i healpixes.\n", il_size(loosen));
					fflush(stdout);

					newlist = il_new(1024);

					for (i=0; i<il_size(loosen); i++) {
						int N;
						hp = il_get(loosen, i);

						if (!find_stars_and_vectors(hp, Nside, radius2,
													dxfrac, dyfrac,
													NULL, NULL, NULL, NULL, NULL,
													&N, centre, vx, vy,
													boxx, boxy,
													NULL, mx, dimquads)) {
							il_append(newlist, hp);
							continue;
						}
						if (!create_quad(stars, inds, N, circle,
										 centre, vx, vy, boxx, boxy, TRUE, dimquads)) {
							il_append(newlist, hp);
							continue;
						}
						nmade++;
					}
					il_free(loosen);
					if (!il_size(newlist)) {
						il_free(newlist);
						newlist = NULL;
					}
					loosenhps[xpass * ypasses + ypass] = newlist;

					printf("Made %i quads.\n", nmade);
					printf("Merging quads...\n");
					fflush(stdout);
					for (i=0; i<Nquads; i++) {
						quad* q = quadlist + i;
						bt_insert(bigquadlist, q, FALSE, compare_quads);
					}
					Nquads = 0;
				}
			}
			if (alldone) {
				printf("Made quads in all healpixes - no need to loosen further.\n");
				break;
			}
		}
	}

	if (failedrdls) {
		dl_free(nostars_radec);
		dl_free(noreuse_radec);
		dl_free(noquads_radec);
	}
	if (noreuse_hps)
		il_free(noreuse_hps);

	free(cq_pquads);
	free(cq_inbox);

	free(stars);
	free(inds);
	free(perm);
	free(nuses);

	printf("Writing quads...\n");

	// add the quads from the big-quadlist
	nquads = bt_size(bigquadlist);
	for (i=0; i<nquads; i++) {
		quad* q = bt_access(bigquadlist, i);
		write_quad(codes, quads, q, dimquads, dimcodes);
	}
	// add the quads that were made during the final round.
	for (i=0; i<Nquads; i++) {
		quad* q = quadlist + i;
		write_quad(codes, quads, q, dimquads, dimcodes);
	}
	free(quadlist);

	startree_close(starkd);

	// fix output file headers.
	if (quadfile_fix_header(quads) ||
		quadfile_close(quads)) {
		fprintf(stderr, "Couldn't write quad output file: %s\n", strerror(errno));
		exit( -1);
	}
	if (codefile_fix_header(codes) ||
		codefile_close(codes)) {
		fprintf(stderr, "Couldn't write code output file: %s\n", strerror(errno));
		exit( -1);
	}

	if (failedrdls) {
		if (rdlist_fix_primary_header(failedrdls) ||
			rdlist_close(failedrdls)) {
			fprintf(stderr, "Failed to fix header of failed RDLS file.\n");
		}
	}

	bt_free(bigquadlist);

	if (loosenmax) {
        for (i=0; i<xpasses*ypasses; i++)
            il_free(loosenhps[i]);
		free(loosenhps);
    }

	toc();
	printf("Done.\n");
	return 0;
}

	
