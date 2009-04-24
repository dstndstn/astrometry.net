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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "kdtree.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "matchobj.h"
#include "catalog.h"
#include "tic.h"
#include "quadfile.h"
#include "intmap.h"
#include "xylist.h"
#include "rdlist.h"
#include "qidxfile.h"
#include "verify.h"
#include "qfits.h"
#include "ioutils.h"
#include "starkd.h"
#include "codekd.h"
#include "index.h"
#include "qidxfile.h"
#include "boilerplate.h"
#include "sip.h"
#include "sip_qfits.h"
#include "log.h"
#include "fitsioutils.h"
#include "blind_wcs.h"
#include "codefile.h"
#include "solver.h"

const char* OPTIONS = "hx:w:i:v"; //q:";

void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
		   "   -w <WCS input file>\n"
		   "   -x <xyls input file>\n"
		   "   -i <index-name>\n"
		   //"   [-q <qidx-name>]\n"
           "   -v: verbose\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
	int c;
	char* xylsfn = NULL;
	char* wcsfn = NULL;

	sl* indexnames;
	pl* indexes;
	pl* qidxes;

	xylist_t* xyls = NULL;
	sip_t sip;
	int i;
	int W, H;
	double xyzcenter[3];
	double fieldrad2;

	double pixr2 = 1.0;

    int loglvl = LOG_MSG;

	fits_use_error_system();

	indexnames = sl_new(8);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'i':
			sl_append(indexnames, optarg);
			break;
		case 'x':
			xylsfn = optarg;
			break;
		case 'w':
			wcsfn = optarg;
			break;
        case 'v':
            loglvl++;
            break;
		}
	}
	if (optind != argc) {
		print_help(args[0]);
		exit(-1);
	}
	if (!xylsfn || !wcsfn) {
		print_help(args[0]);
		exit(-1);
	}
    log_init(loglvl);

	// read WCS.
	logmsg("Trying to parse SIP header from %s...\n", wcsfn);
	if (!sip_read_header_file(wcsfn, &sip)) {
		logmsg("Failed to parse SIP header from %s.\n", wcsfn);
	}
	// image W, H
	W = sip.wcstan.imagew;
	H = sip.wcstan.imageh;
	if ((W == 0.0) || (H == 0.0)) {
		logmsg("WCS file %s didn't contain IMAGEW and IMAGEH headers.\n", wcsfn);
		// FIXME - use bounds of xylist?
		exit(-1);
	}

	// read XYLS.
	xyls = xylist_open(xylsfn);
	if (!xyls) {
		logmsg("Failed to read an xylist from file %s.\n", xylsfn);
		exit(-1);
	}

	// read indices.
	indexes = pl_new(8);
	qidxes = pl_new(8);
	for (i=0; i<sl_size(indexnames); i++) {
		char* name = sl_get(indexnames, i);
		index_t* indx;
		char* qidxfn;
		qidxfile* qidx;
		logmsg("Loading index from %s...\n", name);
		indx = index_load(name, 0);
		if (!indx) {
			logmsg("Failed to read index \"%s\".\n", name);
			exit(-1);
		}
		pl_append(indexes, indx);

		logmsg("Index name: %s\n", indx->meta.indexname);

        qidxfn = index_get_qidx_filename(indx->meta.indexname);
		qidx = qidxfile_open(qidxfn);
		if (!qidx) {
			logmsg("Failed to open qidxfile \"%s\".\n", qidxfn);
            exit(-1);            
		}
		free(qidxfn);
		pl_append(qidxes, qidx);
	}
	sl_free2(indexnames);

	// Find field center and radius.
	sip_pixelxy2xyzarr(&sip, W/2, H/2, xyzcenter);
	fieldrad2 = arcsec2distsq(sip_pixel_scale(&sip) * hypot(W/2, H/2));

	// Find all stars in the field.
	for (i=0; i<pl_size(indexes); i++) {
		kdtree_qres_t* res;
		index_t* indx;
		int nquads;
		uint32_t* quads;
		int j;
		qidxfile* qidx;
		il* uniqquadlist;

        // index stars that are inside the image.
		il* starlist;

        // quads that are at least partly-contained in the image.
		il* quadlist;

        // quads that are fully-contained in the image.
		il* fullquadlist;

        // index stars that are in partly-contained quads.
		il* starsinquadslist;

        // index stars that are in fully-contained quads.
		il* starsinquadsfull;

        // index stars that are in quads and have correspondences.
		il* corrstars;
        // the corresponding field stars
        il* corrfield;

        // quads that are fully in the image and built from stars with correspondences.
		il* corrfullquads;


		dl* starxylist;
		il* corrquads;
		il* corruniqquads;
        starxy_t* xy;

        // (x,y) positions of field stars.
		double* fieldxy;

		int Nfield;
		kdtree_t* ftree;
		int Nleaf = 5;
        int dimquads, dimcodes;

		indx = pl_get(indexes, i);
		qidx = pl_get(qidxes, i);

		// Find index stars.
		res = kdtree_rangesearch_options(indx->starkd->tree, xyzcenter, fieldrad2*1.05,
										 KD_OPTIONS_SMALL_RADIUS | KD_OPTIONS_RETURN_POINTS);
		if (!res || !res->nres) {
			logmsg("No index stars found.\n");
			exit(-1);
		}
		logmsg("Found %i index stars in range.\n", res->nres);

		starlist = il_new(16);
		starxylist = dl_new(16);

		// Find which ones in range are inside the image rectangle.
		for (j=0; j<res->nres; j++) {
			int starnum = res->inds[j];
			double x, y;
			if (!sip_xyzarr2pixelxy(&sip, res->results.d + j*3, &x, &y))
				continue;
			if ((x < 0) || (y < 0) || (x >= W) || (y >= H))
				continue;
			il_append(starlist, starnum);
			dl_append(starxylist, x);
			dl_append(starxylist, y);
		}
		logmsg("Found %i index stars inside the field.\n", il_size(starlist));

		uniqquadlist = il_new(16);
		quadlist = il_new(16);

		// For each index star, find the quads of which it is a part.
		for (j=0; j<il_size(starlist); j++) {
			int k;
			int starnum = il_get(starlist, j);
			if (qidxfile_get_quads(qidx, starnum, &quads, &nquads)) {
				logmsg("Failed to get quads for star %i.\n", starnum);
				exit(-1);
			}
			//logmsg("star %i is involved in %i quads.\n", starnum, nquads);
			for (k=0; k<nquads; k++) {
				il_insert_ascending(quadlist, quads[k]);
				il_insert_unique_ascending(uniqquadlist, quads[k]);
			}
		}
		logmsg("Found %i quads partially contained in the field.\n", il_size(uniqquadlist));

		// Find quads that are fully contained in the image.
		fullquadlist = il_new(16);
		for (j=0; j<il_size(uniqquadlist); j++) {
			int quad = il_get(uniqquadlist, j);
			int ind = il_index_of(quadlist, quad);
			if (ind + 3 >= il_size(quadlist))
				continue;
			if (il_get(quadlist, ind+3) != quad)
				continue;
			il_append(fullquadlist, quad);
		}
		logmsg("Found %i quads fully contained in the field.\n", il_size(fullquadlist));

        dimquads = quadfile_dimquads(indx->quads);
        dimcodes = dimquad2dimcode(dimquads);

		// Find the stars that are in quads.
		starsinquadslist = il_new(16);
		for (j=0; j<il_size(uniqquadlist); j++) {
            int k;
			unsigned int stars[dimquads];
			int quad = il_get(uniqquadlist, j);
			quadfile_get_stars(indx->quads, quad, stars);
            for (k=0; k<dimquads; k++)
                il_insert_unique_ascending(starsinquadslist, stars[k]);
		}
		logmsg("Found %i stars involved in quads (partially contained).\n", il_size(starsinquadslist));

		// Find the stars that are in quads that are completely contained.
		starsinquadsfull = il_new(16);
		for (j=0; j<il_size(fullquadlist); j++) {
            int k;
			unsigned int stars[dimquads];
			int quad = il_get(fullquadlist, j);
			quadfile_get_stars(indx->quads, quad, stars);
            for (k=0; k<dimquads; k++)
                il_insert_unique_ascending(starsinquadsfull, stars[k]);
		}
		logmsg("Found %i stars involved in quads (fully contained).\n", il_size(starsinquadsfull));

		// Now find correspondences between index objects and field objects.
        xy = xylist_read_field(xyls, NULL);
        if (!xy) {
			logmsg("Failed to read xyls entries.\n");
			exit(-1);
        }
        Nfield = starxy_n(xy);
        fieldxy = starxy_to_flat_array(xy, NULL);
        //starxy_free(xy);

		// Build a tree out of the field objects (in pixel space)
        // NOTE that fieldxy is permuted in this process!
        {
            double* fxycopy = malloc(Nfield * 2 * sizeof(double));
            memcpy(fxycopy, fieldxy, Nfield * 2 * sizeof(double));
            //ftree = kdtree_build(NULL, fieldxy, Nfield, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);
            ftree = kdtree_build(NULL, fxycopy, Nfield, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);
        }
		if (!ftree) {
			logmsg("Failed to build kdtree.\n");
			exit(-1);
		}
		// For each index object involved in quads, search for a correspondence.
		corrstars = il_new(16);
        corrfield = il_new(16);
		for (j=0; j<il_size(starsinquadslist); j++) {
			int star;
			double sxyz[3];
			double sxy[2];
			kdtree_qres_t* fres;
			star = il_get(starsinquadslist, j);
			if (startree_get(indx->starkd, star, sxyz)) {
				logmsg("Failed to get position for star %i.\n", star);
				exit(-1);
			}
			if (!sip_xyzarr2pixelxy(&sip, sxyz, sxy, sxy+1)) {
				logmsg("SIP backward for star %i.\n", star);
				exit(-1);
			}
			fres = kdtree_rangesearch_options(ftree, sxy, pixr2,
											  KD_OPTIONS_SMALL_RADIUS);
			if (!fres || !fres->nres)
				continue;
			if (fres->nres > 1) {
				logmsg("%i matches for star %i.\n", fres->nres, star);
			}

			il_append(corrstars, star);
            il_append(corrfield, fres->inds[0]); //kdtree_permute(ftree, fres->inds[0]));

            /*{
              double fx, fy;
              int fi;
              fi = il_get(corrfield, il_size(corrfield)-1);
              fx = fieldxy[2*fi + 0];
              fy = fieldxy[2*fi + 1];
              logmsg("star   %g,%g\n", sxy[0], sxy[1]);
              logmsg("field  %g,%g\n", fx, fy);
              }*/
		}
		logmsg("Found %i correspondences for stars involved in quads (partially contained).\n",
				il_size(corrstars));

		// Find quads built only from stars with correspondences.
		corrquads = il_new(16);
		corruniqquads = il_new(16);
		for (j=0; j<il_size(corrstars); j++) {
			int k;
			int starnum = il_get(corrstars, j);
			if (qidxfile_get_quads(qidx, starnum, &quads, &nquads)) {
				logmsg("Failed to get quads for star %i.\n", starnum);
				exit(-1);
			}
			for (k=0; k<nquads; k++) {
				il_insert_ascending(corrquads, quads[k]);
				il_insert_unique_ascending(corruniqquads, quads[k]);
			}
		}

		// Find quads that are fully contained in the image.
		corrfullquads = il_new(16);
		for (j=0; j<il_size(corruniqquads); j++) {
			int quad = il_get(corruniqquads, j);
			int ind = il_index_of(corrquads, quad);
			if (ind + 3 >= il_size(corrquads))
				continue;
			if (il_get(corrquads, ind+3) != quad)
				continue;
			il_append(corrfullquads, quad);
		}
		logmsg("Found %i quads built from stars with correspondencs, fully contained in the field.\n", il_size(corrfullquads));

		for (j=0; j<il_size(corrfullquads); j++) {
			unsigned int stars[dimquads];
			int k;
			int ind;
			//double px,py;
            double starxyz[3 * dimquads];
            double starxy[2 * dimquads];

            double realcode[dimcodes];
            //double starcode[dimcodes];
            double fieldcode[dimcodes];

            tan_t wcs;

            MatchObj mo;

			int quad = il_get(corrfullquads, j);

            memset(&mo, 0, sizeof(MatchObj));

			quadfile_get_stars(indx->quads, quad, stars);

			/*
              logmsg("quad #%i: quad id %i.  stars", j, quad);
              for (k=0; k<dimquads; k++)
			  logmsg(" %i", stars[k]);
              logmsg("\n");
            */

            codetree_get(indx->codekd, quad, realcode);

            for (k=0; k<dimquads; k++) {
                int find;
                // position of corresponding field star.
                ind = il_index_of(corrstars, stars[k]);
                assert(ind >= 0);
                find = il_get(corrfield, ind);
                starxy[k*2 + 0] = fieldxy[find*2 + 0];
                starxy[k*2 + 1] = fieldxy[find*2 + 1];
                // index star xyz.
                startree_get(indx->starkd, stars[k], starxyz + 3*k);

                mo.star[k] = stars[k];
                mo.field[k] = find;
                /*
                  {
                  double sx, sy;
                  sip_xyzarr2pixelxy(&sip, starxyz + 3*k, &sx, &sy);
                  logmsg("star  %g,%g\n", sx, sy);
                  logmsg("field %g,%g\n", starxy[k*2+0], starxy[k*2+1]);
                  }
                */
            }

			logmsg("quad #%i: stars\n", j);
            for (k=0; k<dimquads; k++)
                logmsg(" %i", mo.field[k]);
            logmsg("\n");


            codefile_compute_field_code(starxy, fieldcode, dimquads);
            //codefile_compute_star_code (starxyz, starcode, dimquads);

            /*
              logmsg("real code:");
              for (k=0; k<dimcodes; k++)
              logmsg(" %g", realcode[k]);
              logmsg("\n");
              logmsg("star code:");
              for (k=0; k<dimcodes; k++)
              logmsg(" %g", starcode[k]);
              logmsg("\n");
              logmsg("field code:");
              for (k=0; k<dimcodes; k++)
              logmsg(" %g", fieldcode[k]);
              logmsg("\n");
              logmsg("code distances: %g, %g\n",
              sqrt(distsq(realcode, starcode, dimcodes)),
              sqrt(distsq(realcode, fieldcode, dimcodes)));
            */
            logmsg("  code distance: %g\n",
                    sqrt(distsq(realcode, fieldcode, dimcodes)));

            blind_wcs_compute(starxyz, starxy, dimquads, &wcs, NULL);

            {
                double llxyz[3];
                verify_field_t* vf;
                double verpix2 = DEFAULT_VERIFY_PIX;

                mo.dimquads = dimquads;

                sip_pixelxy2xyzarr(&sip, sip.wcstan.imagew, sip.wcstan.imageh, mo.center);
                sip_pixelxy2xyzarr(&sip, 0, 0, llxyz);
                mo.radius = sqrt(distsq(mo.center, llxyz, 3));

                memcpy(&mo.wcstan, &wcs, sizeof(tan_t));
                mo.wcs_valid = TRUE;

                vf = verify_field_preprocess(xy);

                verify_hit(indx->starkd, indx->meta.cutnside, &mo, NULL, vf, verpix2,
                           DEFAULT_DISTRACTOR_RATIO, sip.wcstan.imagew, sip.wcstan.imageh,
                           log(-1e100), HUGE_VAL, HUGE_VAL, TRUE, FALSE);

                verify_field_free(vf);
            }


            logmsg("Verify odds: %g\n", exp(mo.logodds));


			// Gah! map...
            /*
              for (k=0; k<dimquads; k++) {
              ind = il_index_of(starlist, stars[k]);
              px = dl_get(starxylist, ind*2+0);
              py = dl_get(starxylist, ind*2+1);
              printf("%g %g ", px, py);
              }
              printf("\n");
            */
		}

		il_free(fullquadlist);
		il_free(uniqquadlist);
		il_free(quadlist);
		il_free(starlist);
	}

	if (xylist_close(xyls)) {
		logmsg("Failed to close XYLS file.\n");
	}
	return 0;
}
