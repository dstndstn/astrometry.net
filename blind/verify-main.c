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

/**
 Runs the verification procedure in stand-alone mode.
 */

#include "matchfile.h"
#include "matchobj.h"
#include "index.h"
#include "verify.h"
#include "xylist.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hvi:m:f:";


static void print_help(const char* progname) {
	printf("Usage:   %s [options]\n"
	       "   [-m <match-file>]\n"
		   "   [-i <index-file>]\n"
		   "   [-f <xylist-file>]\n"
	       "\n", progname);
}

int main(int argc, char** args) {
	int argchar;
    bool help = FALSE;
	int loglvl = LOG_MSG;
	char* indexfn = NULL;
	char* matchfn = NULL;
	char* xyfn = NULL;

	index_t* index;
	matchfile* mf;
	MatchObj* mo;
	verify_field_t* vf;
	starxy_t* fieldxy;
	xylist_t* xyls;

	double pix2 = 1.0;
	double distractors = 0.25;
	double fieldW, fieldH;
	double logbail = -1e100;
	bool growvariance = TRUE;
	bool fake = FALSE;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
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

	if (!indexfn || !matchfn || !xyfn) {
		logerr("You must specify index (-i) and matchfile (-m) and xylist (-x).\n");
		print_help(args[0]);
		exit(-1);
	}

	mf = matchfile_open(matchfn);
	if (!mf) {
		ERROR("Failed to read match file %s", matchfn);
		exit(-1);
	}

	index = index_load(indexfn, 0);
	if (!index) {
		ERROR("Failed to open index %s", indexfn);
		exit(-1);
	}

	xyls = xylist_open(xyfn);
	if (!xyls) {
		ERROR("Failed to open xylist %s", xyfn);
		exit(-1);
	}

	mo = matchfile_read_match(mf);
	if (!mo) {
		ERROR("Failed to read object from match file.");
		exit(-1);
	}

	fieldxy = xylist_read_field(xyls, NULL);
	if (!fieldxy) {
		ERROR("Failed to read a field from xylist %s", xyfn);
		exit(-1);
	}

	vf = verify_field_preprocess(fieldxy);

	verify_hit(index->starkd, mo, NULL, vf,
			   pix2, distractors, fieldW, fieldH,
			   logbail, growvariance,
			   index_get_quad_dim(index), fake);

	verify_field_free(vf);

	starxy_free(fieldxy);

	xylist_close(xyls);
	index_close(index);
	matchfile_close(mf);

	return 0;
}
