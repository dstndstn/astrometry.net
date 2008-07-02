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

#include <math.h>
#include "mathutil.h"
#include "starutil.h"
#include "kdtree.h"
#include "kdtree_fits_io.h"

#define OPTIONS "hpn:s:z:f:o:w:x:q:r:d:S:"
const char HelpString[] =
    "genfields -f fname -o fieldname [-F] {-n num_rand_fields | -r RA -d DEC}\n"
    "          -s scale(arcmin) [-p] [-w noise] [-x distractors] [-q dropouts] [-S seed]\n\n"
    "    -r RA -d DEC generates a single field centred at RA,DEC\n"
    "    -n N generates N randomly centred fields\n"
    "    -p flips parity, -q (default 0) sets the fraction of real stars removed\n"
    "    -x (default 0) sets the fraction of real stars added as random stars\n"
    "    -w (default 0) sets the fraction of scale by which to jitter positions\n"
"    -S random seed\n";

extern char *optarg;
extern int optind, opterr, optopt;

static void fopenout(char* fn, FILE** pfid) {
	FILE* fid = fopen(fn, "wb");
	if (!fid) {
		fprintf(stderr, "Error opening file %s: %s\n", fn, strerror(errno));
		exit(-1);
	}
	*pfid = fid;
}

uint gen_pix(FILE *listfid, FILE *pix0fid, FILE *pixfid,
             kdtree_t* starkd,
             double aspect, double noise, double distractors, double dropouts,
             double ramin, double ramax, double decmin, double decmax,
             double radscale, uint numFields);

char *treefname = NULL, *listfname = NULL, *pix0fname = NULL, *pixfname = NULL;
char *rdlsfname = NULL;
FILE *rdlsfid = NULL;
char FlipParity = 0;

int RANDSEED = 777;
/* Number of times a field can fail to be created before bailing */
int FAILURES = 30;

int main(int argc, char *argv[])
{
	int argidx, argchar;
	uint numFields = 0;
	double radscale = 10.0, aspect = 1.0, distractors = 0.0, dropouts = 0.0, noise = 0.0;
	double centre_ra = 0.0, centre_dec = 0.0;
	FILE *listfid = NULL, *pix0fid = NULL, *pixfid = NULL;
	uint numtries;
	uint numstars;
	uint i;
	kdtree_t* starkd = NULL;
    char* basename = NULL;

	if (argc <= 8) {
		fprintf(stderr, HelpString);
		return (HELP_ERR);
	}

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'S':
			RANDSEED = atoi(optarg);
			break;
		case 'n':
			numFields = strtoul(optarg, NULL, 0);
			break;
		case 'p':
			FlipParity = 1;
			break;
		case 's':
			radscale = (double)strtod(optarg, NULL);
			radscale = arcmin2rad(radscale);
			break;
		case 'z':
			aspect = strtod(optarg, NULL);
			break;
		case 'w':
			noise = strtod(optarg, NULL);
			break;
		case 'x':
			distractors = strtod(optarg, NULL);
			break;
		case 'q':
			dropouts = strtod(optarg, NULL);
			break;
		case 'r':
			centre_ra = deg2rad(strtod(optarg, NULL));
			break;
		case 'd':
			centre_dec = deg2rad(strtod(optarg, NULL));
			break;
		case 'f':
			basename = optarg;
			break;
		case 'o':
			listfname = mk_idlistfn(optarg);
			pix0fname = mk_field0fn(optarg);
			pixfname = mk_fieldfn(optarg);
			rdlsfname = mk_rdlsfn(optarg);
			break;
		case '?':
			fprintf(stderr, "Unknown option `-%c'.\n", optopt);
		case 'h':
			fprintf(stderr, HelpString);
			return (HELP_ERR);
		default:
			return (OPT_ERR);
		}

	if (optind < argc) {
		for (argidx = optind; argidx < argc; argidx++)
			fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
		fprintf(stderr, HelpString);
		return (OPT_ERR);
	}
    if (!basename) {
		fprintf(stderr, HelpString);
        exit(-1);
    }

	srand(RANDSEED);

	if (numFields)
		fprintf(stderr, "genfields: making %u random fields from %s [RANDSEED=%d]\n",
		        numFields, treefname, RANDSEED);
	else {
		fprintf(stderr, "genfields: making fields from %s around ra=%lf,dec=%lf\n",
		        treefname, centre_ra, centre_dec);
		numFields = 1;
	}

	treefname = mk_streefn(basename);

	fprintf(stderr, "  Reading star KD tree from %s...", treefname);
	fflush(stderr);

	starkd = kdtree_fits_read_file(treefname);

	free_fn(treefname);
	if (!starkd) {
		fprintf(stderr, "Couldn't read star kdtree.\n");
		exit(-1);
	}
	numstars = starkd->ndata;
	fprintf(stderr, "done\n    (%u stars, %d nodes).\n",
	        numstars, starkd->nnodes);

	if (numFields > 1)
		fprintf(stderr, "  Generating %u fields at scale %g arcmin...\n",
		        numFields, rad2arcmin(radscale));
	fflush(stderr);
	fopenout(listfname, &listfid);
	free_fn(listfname);
	fopenout(pix0fname, &pix0fid);
	free_fn(pix0fname);
	fopenout(pixfname, &pixfid);
	free_fn(pixfname);
	fopenout(rdlsfname, &rdlsfid);
	free_fn(rdlsfname);
	if (numFields > 1) {
		double ramin, ramax, decmin, decmax;
		decmin = ramin = HUGE_VAL;
		decmax = ramax = -HUGE_VAL;
		for (i=0; i<starkd->ndata; i++) {
			double ra, dec;
			double* xyz = starkd->data + i*DIM_STARS;
			ra = xy2ra(xyz[0], xyz[1]);
			dec = z2dec(xyz[2]);
			if (ra > ramax) ramax = ra;
			if (ra < ramin) ramin = ra;
			if (dec > decmax) decmax = dec;
			if (dec < decmin) decmin = dec;
		}
		numtries = gen_pix(listfid, pix0fid, pixfid, starkd, aspect,
		                   noise, distractors, dropouts,
		                   ramin, ramax, decmin, decmax, radscale, numFields);
	} else
		numtries = gen_pix(listfid, pix0fid, pixfid, starkd, aspect,
		                   noise, distractors, dropouts,
		                   centre_ra, centre_ra, centre_dec, centre_dec,
		                   radscale, numFields);
	fclose(listfid);
	fclose(pix0fid);
	fclose(pixfid);
	fclose(rdlsfid);
	if (numFields > 1)
		fprintf(stderr, "  made %u nonempty fields in %u tries.\n",
		        numFields, numtries);

	kdtree_close(starkd);
	return 0;
}



uint gen_pix(FILE *listfid, FILE *pix0fid, FILE *pixfid,
             kdtree_t* starkd,
             double aspect, double noise, double distractors, double dropouts,
             double ramin, double ramax, double decmin, double decmax,
             double radscale, uint numFields)
{
	uint jj, numS, numX;
	uint numtries = 0, ii;
	double xx, yy;
	double scale = sqrt(arc2distsq(radscale));
	double pixxmin = 0, pixymin = 0, pixxmax = 0, pixymax = 0;
	double randstar[3];
	kdtree_qres_t* krez = NULL;

	fprintf(pix0fid, "NumFields=%u\n", numFields);
	fprintf(pixfid, "NumFields=%u\n", numFields);
	fprintf(listfid, "NumFields=%u\n", numFields);
	fprintf(rdlsfid, "NumFields=%u\n", numFields);

	for (ii = 0;ii < numFields;ii++) {
		numS = 0;
		while (!numS) {
			make_rand_star(randstar, ramin, ramax, decmin, decmax);

			krez = kdtree_rangesearch(starkd, randstar, scale*scale);

			numS = krez->nres;
			//fprintf(stderr,"random location: %u within scale.\n",numS);

			if (numS) {
				fprintf(pix0fid, "centre xyz=(%lf,%lf,%lf) radec=(%lf,%lf)\n",
						randstar[0], randstar[1], randstar[2],
						rad2deg(xy2ra(randstar[0], randstar[1])),
						rad2deg(z2dec(randstar[2])));

				numX = floor(numS * distractors);
				fprintf(pixfid, "%u", numS + numX);
				fprintf(pix0fid, "%u", numS);
				fprintf(listfid, "%u", numS);
				fprintf(rdlsfid, "%u", numS);
				for (jj = 0; jj < numS; jj++) {
					double x, y, z, ra, dec;
					fprintf(listfid, ",%d", krez->inds[jj]);
					star_coords(krez->results + jj*DIM_STARS,
								randstar, &xx, &yy);

					x = krez->results[jj*DIM_STARS + 0];
					y = krez->results[jj*DIM_STARS + 1];
					z = krez->results[jj*DIM_STARS + 2];
					ra = rad2deg(xy2ra(x,y));
					dec = rad2deg(z2dec(z));
					fprintf(rdlsfid, ",%lf,%lf", ra, dec);

					// should add random rotation here ???
					if (FlipParity) {
						double swaptmp = xx;
						xx = yy;
						yy = swaptmp;
					}
					if (jj == 0) {
						pixxmin = pixxmax = xx;
						pixymin = pixymax = yy;
					}
					if (xx > pixxmax)
						pixxmax = xx;
					if (xx < pixxmin)
						pixxmin = xx;
					if (yy > pixymax)
						pixymax = yy;
					if (yy < pixymin)
						pixymin = yy;
					fprintf(pix0fid, ",%lf,%lf", xx, yy);
					if (noise)
						fprintf(pixfid, ",%lf,%lf",
						        gaussian_sample(xx, noise*scale),
						        gaussian_sample(yy, noise*scale));
					else
						fprintf(pixfid, ",%lf,%lf", xx, yy);
				}
				for (jj = 0;jj < numX;jj++)
					fprintf(pixfid, ",%lf,%lf",
							uniform_sample(pixxmin, pixxmax),
							uniform_sample(pixymin, pixymax));

				fprintf(pixfid, "\n");
				fprintf(listfid, "\n");
				fprintf(pix0fid, "\n");
			}
			kdtree_free_query(krez);
			numtries++;

			if (numtries >= FAILURES) {
				/* We've failed too many times; something is wrong. Bail
				 * gracefully. */
				fprintf(stderr, "  ERROR: Too many failures: %u fails\n",
					numtries);
				exit(1);
			}
		}
	}
	return numtries;
}
