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

#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"
#include "wcs-xy2rd.h"
#include "bl.h"
#include "ioutils.h"
#include "rdlist.h"
#include "kdtree.h"
#include "allquads.h"
#include "sip.h"
#include "sip_qfits.h"
#include "codefile.h"
#include "codekd.h"
#include "unpermute-quads.h"
#include "unpermute-stars.h"
#include "merge-index.h"
#include "fitsioutils.h"

const char* OPTIONS = "hvx:w:l:u:o:d:I:n:";

static void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
	       "      -x <input-xylist>    input: source positions; assumed sorted by brightness\n"
		   "      -w <input-wcs>       input: WCS file for sources\n"
		   "      -o <output-index>    output filename for index\n"
		   "      [-l <min-quad-size>]: minimum fraction of the image size (diagonal) to make quads (default 0.05)\n"
		   "      [-u <max-quad-size>]: maximum fraction of the image size (diagonal) to make quads (default 1.0)\n"
		   "      [-d <dimquads>] number of stars in a \"quad\" (default 4).\n"
		   "      [-n <n-stars>]: use only the first N stars in the xylist.\n"
		   "      [-I <unique-id>] set the unique ID of this index\n"
		   "\n"
		   "      [-v]: add verbosity.\n"
	       "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** argv) {
	int argchar;

	char* xylsfn = NULL;
	char* wcsfn = NULL;
	char* indexfn = NULL;

	double lowf = 0.1;
	double highf = 1.0;

	int dimquads = 4;
	int loglvl = LOG_MSG;
	int id = 0;
	int i;
	int nstars = 0;

	sl* tempfiles;
	char* tempdir = "/tmp";
	int wcsext = 0;
	char* rdlsfn;
	char* skdtfn;
	char* quadfn;
	char* codefn;
	char* ckdtfn;
	char* skdt2fn;
	char* quad2fn;
	char* quad3fn;
	char* ckdt2fn;

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'n':
			nstars = atoi(optarg);
			break;
		case 'v':
			loglvl++;
			break;
		case 'd':
			dimquads = atoi(optarg);
			break;
		case 'I':
			id = atoi(optarg);
			break;
		case 'h':
			print_help(argv[0]);
			exit(0);
		case 'x':
            xylsfn = optarg;
			break;
		case 'w':
            wcsfn = optarg;
			break;
		case 'o':
			indexfn = optarg;
			break;
		case 'u':
			highf = atof(optarg);
			break;
		case 'l':
			lowf = atof(optarg);
			break;
		default:
			return -1;
		}
	
	log_init(loglvl);

	if (!xylsfn || !wcsfn || !indexfn) {
		printf("Specify in & out filenames, bonehead!\n");
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

	if (!id)
		logmsg("Warning: you should set the unique-id for this index (with -I).\n");

	if (dimquads > DQMAX) {
		ERROR("Quad dimension %i exceeds compiled-in max %i.\n", dimquads, DQMAX);
		exit(-1);
	}

    tempfiles = sl_new(4);

	// wcs-xy2rd
	rdlsfn = create_temp_file("rdls", tempdir);
	sl_append_nocopy(tempfiles, rdlsfn);
	logmsg("Writing RA,Decs to %s\n", rdlsfn);
	if (wcs_xy2rd(wcsfn, wcsext, xylsfn, rdlsfn, NULL, NULL, FALSE, NULL)) {
		ERROR("Failed to convert xylist to rdlist");
		exit(-1);
	}
        
	// startree
	skdtfn = create_temp_file("skdt", tempdir);
	sl_append_nocopy(tempfiles, skdtfn);
	{
		int Nleaf = 25;
		int exttype  = KDT_EXT_DOUBLE;
		int datatype = KDT_DATA_U32;
		int treetype = KDT_TREE_U32;
		int tt;
		int buildopts = KD_BUILD_SPLIT;
		int N, D;
		startree_t* starkd;
		rdlist_t* rdls;
		rd_t* rd;
		double* xyz;
		double low[3];
		double high[3];
		int d;
		rdls = rdlist_open(rdlsfn);
		if (!rdls) {
			ERROR("Failed to open RDLS");
			exit(-1);
		}
		rd = rdlist_read_field(rdls, NULL);
		if (!rd) {
			ERROR("Failed to read RA,Decs");
			exit(-1);
		}
		N = rd_n(rd);
		if (nstars && nstars < N) {
			N = nstars;
		}
		D = 3;

		xyz = malloc(N * D * sizeof(double));
		radecdeg2xyzarrmany(rd->ra, rd->dec, xyz, N);

		rd_free(rd);
		rdlist_close(rdls);

		starkd = startree_new();
		if (!starkd) {
			ERROR("Failed to allocate startree");
			exit(-1);
		}
		tt = kdtree_kdtypes_to_treetype(exttype, treetype, datatype);
		starkd->tree = kdtree_new(N, D, Nleaf);
		for (d=0; d<3; d++) {
			low[d] = -1.0;
			high[d] = 1.0;
		}
		kdtree_set_limits(starkd->tree, low, high);
		logverb("Building star kdtree...\n");
		starkd->tree = kdtree_build(starkd->tree, xyz, N, D, Nleaf, tt, buildopts);
		if (!starkd->tree) {
			ERROR("Failed to build star kdtree");
			exit(-1);
		}
		starkd->tree->name = strdup(STARTREE_NAME);
		logverb("Writing skdt to %s...\n", skdtfn);

		if (startree_write_to_file(starkd, skdtfn)) {
			ERROR("Failed to write star kdtree to %s", skdtfn);
			exit(-1);
		}
		startree_close(starkd);
	}

	// FIXME -- write a temporary skdt containing only the stars we
	// want to index, then a "full" skdt with the stars we want to have
	// available for verifying?

	quadfn = create_temp_file("quad", tempdir);
	sl_append_nocopy(tempfiles, quadfn);
	codefn = create_temp_file("code", tempdir);
	sl_append_nocopy(tempfiles, codefn);
	// allquads
	{
		allquads_t* aq;
		double diagpix, diag;
		sip_t sip;
		qfits_header* hdr;

		// read WCS.
		if (!sip_read_tan_or_sip_header_file_ext(wcsfn, wcsext, &sip, FALSE)) {
			ERROR("Failed to read WCS file %s", wcsfn);
			exit(-1);
		}
		// in pixels
		diagpix = hypot(sip.wcstan.imagew, sip.wcstan.imageh);
		// in arcsec
		diag = diagpix * sip_pixel_scale(&sip);

		logmsg("Image is %i x %i pixels\n", (int)sip.wcstan.imagew, (int)sip.wcstan.imageh);
		logmsg("Setting quad scale range to [%g, %g] pixels, [%g, %g] arcsec\n",
			   diagpix * lowf, diagpix * highf, diag * lowf, diag * highf);

		aq = allquads_init();
		aq->dimquads = dimquads;
		aq->dimcodes = dimquad2dimcode(aq->dimquads);
		aq->id = id;
		aq->quadfn = quadfn;
		aq->codefn = codefn;
		aq->skdtfn = skdtfn;
		aq->quad_d2_lower = arcsec2distsq(diag * lowf);
		aq->quad_d2_upper = arcsec2distsq(diag * highf);
		aq->use_d2_lower = TRUE;
		aq->use_d2_upper = TRUE;

		if (allquads_open_outputs(aq)) {
			exit(-1);
		}
		hdr = codefile_get_header(aq->codes);
		qfits_header_add(hdr, "CIRCLE", "T", "Codes live in the circle", NULL);
		if (allquads_create_quads(aq) ||
			allquads_close(aq)) {
			exit(-1);
		}
		allquads_free(aq);
	}

	// codetree
	ckdtfn = create_temp_file("ckdt", tempdir);
	sl_append_nocopy(tempfiles, ckdtfn);
	{
		int Nleaf = 25;
		codetree *codekd;
		codefile* codes;
		int exttype = KDT_EXT_DOUBLE;
		int datatype = KDT_DATA_U16;
		int treetype = KDT_TREE_U16;
		int tt;
		int buildopts = KD_BUILD_SPLIT;
		int N, D;
		qfits_header* chdr;
		qfits_header* hdr;

		codes = codefile_open(codefn);
		if (!codes) {
			ERROR("Failed to open code file %s", codefn);
			exit(-1);
		}
		N = codes->numcodes;
		logmsg("Read %i codes\n", N);
		codekd = codetree_new();
		if (!codekd) {
			ERROR("Failed to allocate a codetree structure");
			exit(-1);
		}
		chdr = codefile_get_header(codes);
		hdr = codetree_header(codekd);
		fits_header_add_int(hdr, "NLEAF", Nleaf, "Target number of points in leaves.");
		fits_copy_header(chdr, hdr, "INDEXID");
		fits_copy_header(chdr, hdr, "HEALPIX");
		fits_copy_header(chdr, hdr, "HPNSIDE");
		fits_copy_header(chdr, hdr, "CXDX");
		fits_copy_header(chdr, hdr, "CXDXLT1");
		fits_copy_header(chdr, hdr, "CIRCLE");

		tt = kdtree_kdtypes_to_treetype(exttype, treetype, datatype);
		D = codefile_dimcodes(codes);
		codekd->tree = kdtree_new(N, D, Nleaf);
		{
			double low[D];
			double high[D];
			int d;
			for (d=0; d<D; d++) {
				low [d] = 0.5 - M_SQRT1_2;
				high[d] = 0.5 + M_SQRT1_2;
			}
			kdtree_set_limits(codekd->tree, low, high);
		}
		logverb("Building code kdtree...\n");
		codekd->tree = kdtree_build(codekd->tree, codes->codearray, N, D,
									Nleaf, tt, buildopts);
		if (!codekd->tree) {
			ERROR("Failed to build code kdtree");
			exit(-1);
		}
		codekd->tree->name = strdup(CODETREE_NAME);
		logverb("Writing code kdtree to %s...\n", ckdtfn);
		if (codetree_write_to_file(codekd, ckdtfn)) {
			ERROR("Failed to write ckdt to %s", ckdtfn);
			exit(-1);
		}
		codefile_close(codes);
		kdtree_free(codekd->tree);
		codekd->tree = NULL;
		codetree_close(codekd);
	}

	// unpermute-stars
	skdt2fn = create_temp_file("skdt2", tempdir);
	sl_append_nocopy(tempfiles, skdt2fn);
	quad2fn = create_temp_file("quad2", tempdir);
	sl_append_nocopy(tempfiles, quad2fn);
	logmsg("Unpermuting stars from %s and %s to %s and %s\n", skdtfn, quadfn, skdt2fn, quad2fn);
	if (unpermute_stars(skdtfn, quadfn,
						skdt2fn, quad2fn,
						TRUE, FALSE, argv, argc)) {
		ERROR("Failed to unpermute-stars");
		exit(-1);
	}

	// unpermute-quads
	ckdt2fn = create_temp_file("ckdt2", tempdir);
	sl_append_nocopy(tempfiles, ckdt2fn);
	quad3fn = create_temp_file("quad3", tempdir);
	sl_append_nocopy(tempfiles, quad3fn);
	logmsg("Unpermuting quads from %s and %s to %s and %s\n", quad2fn, ckdtfn, quad3fn, ckdt2fn);
	if (unpermute_quads(quad2fn, ckdtfn,
						quad3fn, ckdt2fn, argv, argc)) {
		ERROR("Failed to unpermute-quads");
		exit(-1);
	}

	// mergeindex
	logmsg("Merging %s and %s and %s to %s\n", quad3fn, ckdt2fn, skdt2fn, indexfn);
	if (merge_index(quad3fn, ckdt2fn, skdt2fn, indexfn)) {
		ERROR("Failed to merge-index");
		exit(-1);
	}

	printf("Done.\n");
	return 0;
}

