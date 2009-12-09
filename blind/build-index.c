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
#include "uniformize-catalog.h"
#include "startree2.h"
#include "codetree.h"
#include "bl.h"
#include "ioutils.h"
#include "rdlist.h"
#include "kdtree.h"
#include "hpquads.h"
#include "sip.h"
#include "sip_qfits.h"
#include "codefile.h"
#include "codekd.h"
#include "unpermute-quads.h"
#include "unpermute-stars.h"
#include "merge-index.h"
#include "fitsioutils.h"

const char* OPTIONS = "hvi:o:N:l:u:S:fU:H:s:m:n:r:d:p:R:L:EI:M";

static void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
	       "      -i <input-FITS-catalog>  input: source RA,DEC, etc\n"
		   "      -o <output-index>        output filename for index\n"
		   "      -N <nside>            healpix Nside for quad-building\n"
		   "      -l <min-quad-size>    minimum quad size (arcminutes)\n"
		   "      -u <max-quad-size>    maximum quad size (arcminutes)\n"
		   "      [-S]: sort column (default: assume already sorted)\n"
		   "      [-f]: sort in descending order (eg, for FLUX); default ascending (eg, for MAG)\n"
		   "      [-U]: healpix Nside for uniformization (default: same as -n)\n"
		   "      [-H <big healpix>]; default is all-sky\n"
           "      [-s <big healpix Nside>]; default is 1\n"
		   "      [-m <margin>]: add a margin of <margin> healpixels; default 0\n"
		   "      [-n <sweeps>]    (ie, number of stars per fine healpix grid cell); default 10\n"
		   "      [-r <dedup-radius>]: deduplication radius in arcseconds; default no deduplication\n"
		   "\n"
		   "      [-d <dimquads>] number of stars in a \"quad\" (default 4).\n"
		   "      [-p <passes>]   number of rounds of quad-building (ie, # quads per healpix cell, default 1)\n"
		   "      [-R <reuse-times>] number of times a star can be used.\n"
		   "      [-L <max-reuses>] make extra passes through the healpixes, increasing the \"-r\" reuse\n"
		   "                     limit each time, up to \"max-reuses\".\n"
		   "      [-E]: scan through the catalog, checking which healpixes are occupied.\n"
		   "\n"
		   "      [-I <unique-id>] set the unique ID of this index\n"
		   "\n"
		   "      [-M]: in-memory (don't use temp files)\n"
		   "      [-v]: add verbosity.\n"
	       "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** argv) {
	int argchar;

	char* infn = NULL;
	char* indexfn = NULL;

	// uniformization:
	char* sortcol = NULL;
	bool sortasc = TRUE;
	int bighp = -1;
	int bignside = 1;
	int sweeps = 10;
	double dedup = 0.0;
	int margin = 0;
	int UNside = 0;

	char* racol = "RA";
	char* deccol = "DEC";

	// star kdtree
	startree_t* starkd = NULL;
	fitstable_t* startag = NULL;

	// indexing:
	int Nside = 0;
	double qlo = 0;
	double qhi = 0;
	int passes = 1;
	int Nreuse = 3;
	int Nloosen = 0;
	bool scanoccupied = FALSE;
	int dimquads = 4;

	fitstable_t* catalog;
	fitstable_t* uniform;

	codefile* codes = NULL;
	quadfile* quads = NULL;

	codetree* codekd = NULL;

	int loglvl = LOG_MSG;
	int id = 0;
	int i;

	bool inmemory = FALSE;

	sl* tempfiles;
	char* tempdir = "/tmp";
	char* unifn = NULL;
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
		case 'E':
			scanoccupied = TRUE;
			break;
		case 'L':
			Nloosen = atoi(optarg);
			break;
		case 'R':
			Nreuse = atoi(optarg);
			break;
		case 'p':
			passes = atoi(optarg);
			break;
		case 'M':
			inmemory = TRUE;
			break;
		case 'U':
			UNside = atoi(optarg);
			break;
		case 'i':
            infn = optarg;
			break;
		case 'o':
			indexfn = optarg;
			break;
		case 'u':
			qhi = atof(optarg);
			break;
		case 'l':
			qlo = atof(optarg);
			break;
		case 'S':
			sortcol = optarg;
			break;
		case 'f':
			sortasc = FALSE;
			break;
		case 'H':
			bighp = atoi(optarg);
			break;
		case 's':
			bignside = atoi(optarg);
			break;
		case 'n':
			sweeps = atoi(optarg);
			break;
		case 'N':
			Nside = atoi(optarg);
			break;
		case 'r':
			dedup = atof(optarg);
			break;
		case 'm':
			margin = atoi(optarg);
			break;
		case 'd':
			dimquads = atoi(optarg);
			break;
		case 'I':
			id = atoi(optarg);
			break;
		case 'v':
			loglvl++;
			break;
		case 'h':
			print_help(argv[0]);
			exit(0);
		default:
			return -1;
		}
	
	log_init(loglvl);

	if (!infn || !indexfn) {
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

	logmsg("Reading %s...\n", infn);
	catalog = fitstable_open(infn);
    if (!catalog) {
        ERROR("Couldn't read catalog %s", infn);
        exit(-1);
    }
	logmsg("Got %i stars\n", fitstable_nrows(catalog));

	if (inmemory)
		uniform = fitstable_open_in_memory();
	else {
		unifn = create_temp_file("uniform", tempdir);
		sl_append_nocopy(tempfiles, unifn);
		uniform = fitstable_open_for_writing(unifn);
	}
	if (!uniform) {
		ERROR("Failed to open output table %s", unifn);
		exit(-1);
	}
	if (fitstable_write_primary_header(uniform)) {
		ERROR("Failed to write primary header");
		exit(-1);
	}

	if (!UNside)
		UNside = Nside;

	if (uniformize_catalog(catalog, uniform, racol, deccol,
						   sortcol, sortasc,
						   bighp, bignside, margin,
						   UNside, dedup, sweeps, argv, argc)) {
		exit(-1);
	}

	if (fitstable_fix_primary_header(uniform)) {
		ERROR("Failed to fix output table");
		exit(-1);
	}

	if (!inmemory) {
		if (fitstable_close(uniform)) {
			ERROR("Failed to close output table");
			exit(-1);
		}
	}
	fitstable_close(catalog);

	// startree

	if (inmemory) {
		if (fitstable_switch_to_reading(uniform)) {
			ERROR("Failed to switch uniformized table to read-mode");
			exit(-1);
		}
		startag = fitstable_open_in_memory();

	} else {
		skdtfn = create_temp_file("skdt", tempdir);
		sl_append_nocopy(tempfiles, skdtfn);

		logverb("Reading uniformized catalog %s...\n", unifn);
		uniform = fitstable_open(unifn);
		if (!uniform) {
			ERROR("Failed to open uniformized catalog");
			exit(-1);
		}
	}

	{
		int Nleaf = 25;
		int datatype = KDT_DATA_U32;
		int treetype = KDT_TREE_U32;
		int buildopts = KD_BUILD_SPLIT;

		logverb("Building star kdtree from %i stars\n", fitstable_nrows(uniform));
		starkd = startree_build(uniform, racol, deccol, datatype, treetype,
								buildopts, Nleaf, argv, argc);
		if (!starkd) {
			ERROR("Failed to create star kdtree");
			exit(-1);
		}

		if (!inmemory) {
			logverb("Writing star kdtree to %s\n", skdtfn);
			if (startree_write_to_file(starkd, skdtfn)) {
				ERROR("Failed to write star kdtree");
				exit(-1);
			}
			startree_close(starkd);

			startag = fitstable_open_for_appending(skdtfn);
			if (!startag) {
				ERROR("Failed to re-open star kdtree file %s for appending", skdtfn);
				exit(-1);
			}
		}

		logverb("Adding star kdtree tag-along data...\n");
		if (startree_write_tagalong_table(uniform, startag, racol, deccol)) {
			ERROR("Failed to write tag-along table");
			exit(-1);
		}

		if (fitstable_fix_header(startag)) {
			ERROR("Failed to fix tag-along data header");
			exit(-1);
		}
		if (!inmemory) {
			if (fitstable_close(startag)) {
				ERROR("Failed to close star kdtree tag-along data");
				exit(-1);
			}
		}
	}
	fitstable_close(uniform);

	// hpquads

	if (inmemory) {
		codes = codefile_open_in_memory();
		quads = quadfile_open_in_memory();
		if (hpquads(starkd, codes, quads, Nside,
					qlo, qhi, dimquads, passes, Nreuse, Nloosen,
					id, scanoccupied, argv, argc)) {
			ERROR("hpquads failed");
			exit(-1);
		}
		startree_close(starkd);

	} else {
		quadfn = create_temp_file("quad", tempdir);
		sl_append_nocopy(tempfiles, quadfn);
		codefn = create_temp_file("code", tempdir);
		sl_append_nocopy(tempfiles, codefn);

		if (hpquads_files(skdtfn, codefn, quadfn, Nside,
						  qlo, qhi, dimquads, passes, Nreuse, Nloosen,
						  id, scanoccupied, argv, argc)) {
			ERROR("hpquads failed");
			exit(-1);
		}
	}

	// codetree

	if (inmemory) {
		if (codefile_switch_to_reading(codes)) {
			ERROR("Failed to switch codefile to read-mode");
			exit(-1);
		}
		logmsg("Building code kdtree from %i codes\n", codes->numcodes);
		logmsg("dim: %i\n", codefile_dimcodes(codes));
		codekd = codetree_build(codes, 0, 0, 0, 0, argv, argc);
		if (!codekd) {
			ERROR("Failed to build code kdtree");
			exit(-1);
		}
		if (codefile_close(codes)) {
			ERROR("Failed to close codefile");
			exit(-1);
		}

	} else {

		ckdtfn = create_temp_file("ckdt", tempdir);
		sl_append_nocopy(tempfiles, ckdtfn);

		if (codetree_files(codefn, ckdtfn, 0, 0, 0, 0, argv, argc)) {
			ERROR("codetree failed");
			exit(-1);
		}
	}

	/*
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

	 */

	sl_free2(tempfiles);

	printf("Done.\n");
	return 0;
}

