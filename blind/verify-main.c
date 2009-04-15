/*
 This file is part of the Astrometry.net suite.
 Copyright 2009 Dustin Lang, David W. Hogg.

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

/**
 Runs the verification procedure in stand-alone mode.
 */
#include <sys/param.h>

#include "matchfile.h"
#include "matchobj.h"
#include "index.h"
#include "verify2.h"
#include "xylist.h"
#include "rdlist.h"
#include "log.h"
#include "errors.h"
#include "mathutil.h"

static const char* OPTIONS = "hvi:m:f:r:";

static void print_help(const char* progname) {
	printf("Usage:   %s\n"
	       "   -m <match-file>\n"
		   "   -f <xylist-file>\n"
		   "  (    -i <index-file>\n"
		   "   OR  -r <index-rdls>\n"
		   "  )\n"
           "   [-v]: verbose\n"
	       "\n", progname);
}

static int get_xy_bin(double x, double y,
					  double fieldW, double fieldH,
					  int nw, int nh) {
	int ix, iy;
	ix = (int)floor(nw * x / fieldW);
	ix = MAX(0, MIN(nw-1, ix));
	iy = (int)floor(nh * y / fieldH);
	iy = MAX(0, MIN(nh-1, iy));
	return iy * nw + ix;
}


int main(int argc, char** args) {
	int argchar;
	int loglvl = LOG_MSG;
	char* indexfn = NULL;
	char* matchfn = NULL;
	char* xyfn = NULL;
	char* rdfn = NULL;

	index_t* index = NULL;
	matchfile* mf;
	MatchObj* mo;
	verify_field_t* vf;
	starxy_t* fieldxy;
	xylist_t* xyls;
	rdlist_t* rdls;

	double pix2 = 1.0;
	double distractors = 0.25;
	double fieldW=0, fieldH=0;
	double logbail = log(-1e100);
	double logkeep = log(1e12);
	bool growvariance = TRUE;
	bool fake = FALSE;
	double logodds;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
		case 'r':
			rdfn = optarg;
			break;
		case 'f':
			xyfn = optarg;
			break;
        case 'i':
			indexfn = optarg;
			break;
		case 'm':
			matchfn = optarg;
			break;
		case 'h':
			print_help(args[0]);
			exit(0);
		case 'v':
			loglvl++;
			break;
		}

	log_init(loglvl);

	if (!(indexfn || rdfn) || !matchfn || !xyfn) {
		logerr("You must specify (index (-i) or index rdls (-r)) and matchfile (-m) and field xylist (-f).\n");
		print_help(args[0]);
		exit(-1);
	}

	mf = matchfile_open(matchfn);
	if (!mf) {
		ERROR("Failed to read match file %s", matchfn);
		exit(-1);
	}

	xyls = xylist_open(xyfn);
	if (!xyls) {
		ERROR("Failed to open xylist %s", xyfn);
		exit(-1);
	}
	fieldW = xylist_get_imagew(xyls);
	fieldH = xylist_get_imageh(xyls);

    logmsg("Field W,H = %g, %g\n", fieldW, fieldH);

	mo = matchfile_read_match(mf);
	if (!mo) {
		ERROR("Failed to read object from match file.");
		exit(-1);
	}
    mo->wcstan.imagew = fieldW;
    mo->wcstan.imageh = fieldH;

	fieldxy = xylist_read_field(xyls, NULL);
	if (!fieldxy) {
		ERROR("Failed to read a field from xylist %s", xyfn);
		exit(-1);
	}

	if (indexfn) {
		index = index_load(indexfn, 0);
		if (!index) {
			ERROR("Failed to open index %s", indexfn);
			exit(-1);
		}

		pix2 += square(index->meta.index_jitter / mo->scale);

	} else {
		double indexjitter;

		rdls = rdlist_open(rdfn);
		if (!rdls) {
			ERROR("Failed to open rdlist %s", rdfn);
			exit(-1);
		}

		// HACK
		indexjitter = 1.0; // arcsec.
		pix2 += square(indexjitter / mo->scale);
	}

	logmsg("Pixel jitter: %g pix\n", sqrt(pix2));

	vf = verify_field_preprocess(fieldxy);

	if (index) {
		mo->logodds = 0.0;

		verify_hit(index, mo, NULL, vf,
				   pix2, distractors, fieldW, fieldH,
				   logbail, growvariance,
				   index_get_quad_dim(index), fake,
				   logkeep);

		logodds = mo->logodds;

		index_close(index);

	} else {
		int cutnside;
		int cutnsweeps;
		int indexid;
		int cutnw, cutnh;
		int* cutperm;
		double* testxy;
		double* refxy;
		int i, j, k, NT, NR;
		double* sigma2s;
		rd_t* rd;
		double* bincenters;
		int* binids;
		double effA;

		// -get reference stars
		rd = rdlist_read_field(rdls, NULL);
		if (!rd) {
			ERROR("Failed to read rdls field");
			exit(-1);
		}
		NR = rd_n(rd);
		refxy = malloc(2 * NR * sizeof(double));
		for (i=0; i<NR; i++) {
			double ra, dec;
			ra  = rd_getra (rd, i);
			dec = rd_getdec(rd, i);
			if (!tan_radec2pixelxy(&(mo->wcstan), ra, dec, refxy + 2*i, refxy + 2*i + 1)) {
				ERROR("rdls point projects to wrong side of sphere!");
				exit(-1);
			}
		}
		// -remove the ref star closest to each quad star.
		for (i=0; i<mo->dimquads; i++) {
			double qxy[2];
			int besti = -1;
			double bestd2 = HUGE_VAL;
			if (!tan_xyzarr2pixelxy(&(mo->wcstan), mo->quadxyz + 3*i, qxy, qxy+1)) {
				ERROR("quad star projects to wrong side of sphere!");
				exit(-1);
			}
			for (j=0; j<NR; j++) {
				double d2 = distsq(qxy, refxy + 2*j, 2);
				if (d2 < bestd2) {
					bestd2 = d2;
					besti = j;
				}
			}
			// remove it!
			memmove(refxy + 2*besti, refxy + 2*(besti + 1),
					2*(NR - besti - 1) * sizeof(double));
			NR--;
		}

		logmsg("Reference stars: %i\n", NR);

		// -compute sigma2s
		sigma2s = verify_compute_sigma2s(vf, mo, pix2, !fake);

		// -uniformize field stars
		indexid = mo->indexid;
		if (index_get_missing_cut_params(indexid, &cutnside, &cutnsweeps, NULL, NULL, NULL)) {
			ERROR("Failed to get index cut parameters for index id %i", indexid);
			exit(-1);
		}
		verify_get_uniformize_scale(cutnside, mo->scale, fieldW, fieldH, &cutnw, &cutnh);
		logmsg("Uniformizing test stars into %i x %i bins.\n", cutnw, cutnh);
		verify_uniformize_field(vf, fieldW, fieldH, cutnw, cutnh, &cutperm, NULL, &bincenters, &binids);
		NT = starxy_n(vf->field);

		// get_quad_center
		{
			double Axy[2], Bxy[2], cen[2], Q2;
			starxy_get(vf->field, mo->field[0], Axy);
			starxy_get(vf->field, mo->field[1], Bxy);
			cen[0] = 0.5 * (Axy[0] + Bxy[0]);
			cen[1] = 0.5 * (Axy[1] + Bxy[1]);
			// Find the radius-squared of the quad = distsq(qc, A)
			Q2 = distsq(Axy, cen, 2);
			logmsg("Quad center is (%.1f, %.1f), radius %.1f pix\n", cen[0], cen[1], sqrt(Q2));

			double A = fieldW * fieldH;
			double ror = sqrt(Q2 * (1 + A*(1 - distractors) / (2. * M_PI * NR * pix2)));
			logmsg("Radius of relevance is %.1f\n", ror);

			// Approximate cutting up the image by measuring distance to the bin centers.
			bool* goodbins = malloc(cutnw * cutnh * sizeof(bool));
			int Ngoodbins = 0;

			for (i=0; i<(cutnw * cutnh); i++) {
				double binr = sqrt(distsq(bincenters + 2*i, cen, 2));
				//logmsg("Bin %i (%.1f, %.1f) is %.1f away\n", i, bincenters[2*i], bincenters[2*i+1], binr);
				goodbins[i] = (binr < ror);
				if (goodbins[i])
					Ngoodbins++;
				//if (binr > ror) logmsg("Irrelevant bin!\n");
			}
			// Remove test stars in irrelevant bins...
			k = 0;
			for (i=0; i<NT; i++) {
				if (!goodbins[binids[i]])
					continue;
				cutperm[k] = cutperm[i];
				k++;
			}
			NT = k;
			logmsg("After removing %i/%i irrelevant bins: %i test stars.\n", (cutnw*cutnh)-Ngoodbins, cutnw*cutnh, NT);

			// Effective area: A * proportion of good bins.
			effA = A * Ngoodbins / (double)(cutnw * cutnh);

			// -remove reference stars in bad bins.
			k = 0;
			for (i=0; i<NR; i++) {
				int binid = get_xy_bin(refxy[2*i], refxy[2*i+1], fieldW, fieldH, cutnw, cutnh);
				if (!goodbins[binid])
					continue;
				if (i == k)
					continue;
				memcpy(refxy + 2*k, refxy + 2*i, 2*sizeof(double));
				k++;
			}
			NR = k;
			logmsg("After removing irrelevant ref stars: %i ref stars.\n", NR);

			// New ROR is...
			double newror = sqrt(Q2 * (1 + effA*(1 - distractors) / (2. * M_PI * NR * pix2)));
			logmsg("ROR changed from %g to %g\n", ror, newror);

			free(goodbins);
		}
		free(binids);
		free(bincenters);

		// -remove test quad stars
		testxy = malloc(2 * NT * sizeof(double));
		k = 0;
		for (i=0; i<NT; i++) {
			int starindex = cutperm[i];
			if (!fake) {
				bool inquad = FALSE;
				for (j=0; j<mo->dimquads; j++)
					if (starindex == mo->field[j]) {
						inquad = TRUE;
						break;
					}
				if (inquad)
					continue;
			}
			starxy_get(vf->field, starindex, testxy + 2*k);
			sigma2s[k] = sigma2s[starindex];
			// store their original indices in cutperm so we can look-back.
			//cutperm[k] = starindex;
			k++;
		}
		NT = k;

		logmsg("Test stars: %i\n", NT);

		FILE* f = stderr;

		fprintf(f, "quadxy = array([");
		for (i=0; i<=mo->dimquads; i++)
			fprintf(f, "[%g,%g],", mo->quadpix[2*(i % mo->dimquads)+0], mo->quadpix[2*(i % mo->dimquads)+1]);
		fprintf(f, "])\n");

		fprintf(f, "testxy = array([");
		for (i=0; i<NT; i++)
			fprintf(f, "[%g,%g],", testxy[2*i+0], testxy[2*i+1]);
		fprintf(f, "])\n");

		fprintf(f, "sigmas = array([");
		for (i=0; i<NT; i++)
			fprintf(f, "%g,", sqrt(sigma2s[i]));
		fprintf(f, "])\n");

		fprintf(f, "refxy = array([");
		for (i=0; i<NR; i++)
			fprintf(f, "[%g,%g],", refxy[2*i+0], refxy[2*i+1]);
		fprintf(f, "])\n");

		fprintf(f, "cutx = array([");
		for (i=0; i<=cutnw; i++)
			fprintf(f, "%g,", i * fieldW / (float)cutnw);
		fprintf(f, "])\n");

		fprintf(f, "cuty = array([");
		for (i=0; i<=cutnh; i++)
			fprintf(f, "%g,", i * fieldH / (float)cutnh);
		fprintf(f, "])\n");

		fprintf(f, "W=%i\nH=%i\n", (int)fieldW, (int)fieldH);

		double* all_logodds;
		int* theta;

		logodds = verify_star_lists(refxy, NR, testxy, sigma2s, NT,
									effA, distractors, logbail,
									NULL, NULL, &all_logodds, &theta);

		fprintf(f, "logodds = array([");
		for (i=0; i<NT; i++)
			fprintf(f, "%g,", all_logodds[i]);
		fprintf(f, "])\n");

		fprintf(f, "theta = array([");
		for (i=0; i<NT; i++)
			fprintf(f, "%i,", theta[i]);
		fprintf(f, "])\n");

		free(theta);
		free(all_logodds);
		free(sigma2s);
		free(testxy);
		free(refxy);

		rdlist_close(rdls);
	}
	
    logmsg("Logodds: %g\n", logodds);
    logmsg("Odds: %g\n", logodds);

	verify_field_free(vf);
	starxy_free(fieldxy);

	xylist_close(xyls);
	matchfile_close(mf);

	return 0;
}
