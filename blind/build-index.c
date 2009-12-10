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

#include "build-index.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"
#include "uniformize-catalog.h"
#include "startree2.h"
#include "codetree.h"
#include "unpermute-quads.h"
#include "unpermute-stars.h"
#include "bl.h"
#include "ioutils.h"
#include "rdlist.h"
#include "kdtree.h"
#include "hpquads.h"
#include "sip.h"
#include "sip_qfits.h"
#include "codefile.h"
#include "codekd.h"
#include "merge-index.h"
#include "fitsioutils.h"

int build_index(fitstable_t* catalog, index_params_t* p,
				index_t** p_index, const char* indexfn) {

	fitstable_t* uniform;

	// star kdtree
	startree_t* starkd = NULL;
	fitstable_t* startag = NULL;

	// hpquads
	codefile* codes = NULL;
	quadfile* quads = NULL;

	// codetree
	codetree* codekd = NULL;

	// unpermute-stars
	startree_t* starkd2 = NULL;
	quadfile* quads2 = NULL;
	fitstable_t* startag2 = NULL;

	// unpermute-quads
	quadfile* quads3 = NULL;
	codetree* codekd2 = NULL;

	index_t* index = NULL;

	sl* tempfiles;
	char* unifn;
	char* skdtfn;
	char* quadfn;
	char* codefn;
	char* ckdtfn;
	char* skdt2fn;
	char* quad2fn;
	char* quad3fn;
	char* ckdt2fn;

	if (!p->UNside)
		p->UNside = p->Nside;

	assert(p->Nside);

	if (p->inmemory && !p_index) {
		ERROR("If you set inmemory, you must set p_index");
		return -1;
	}
	if (!p->inmemory && !indexfn) {
		ERROR("If you set !inmemory, you must set indexfn");
		return -1;
	}

    tempfiles = sl_new(4);

	if (p->inmemory)
		uniform = fitstable_open_in_memory();
	else {
		unifn = create_temp_file("uniform", p->tempdir);
		sl_append_nocopy(tempfiles, unifn);
		uniform = fitstable_open_for_writing(unifn);
	}
	if (!uniform) {
		ERROR("Failed to open output table %s", unifn);
		return -1;
	}
	if (fitstable_write_primary_header(uniform)) {
		ERROR("Failed to write primary header");
		return -1;
	}

	if (uniformize_catalog(catalog, uniform, p->racol, p->deccol,
						   p->sortcol, p->sortasc,
						   p->bighp, p->bignside, p->margin,
						   p->UNside, p->dedup, p->sweeps, p->args, p->argc)) {
		return -1;
	}

	if (fitstable_fix_primary_header(uniform)) {
		ERROR("Failed to fix output table");
		return -1;
	}

	if (p->inmemory) {
		if (fitstable_switch_to_reading(uniform)) {
			ERROR("Failed to switch uniformized table to read-mode");
			return -1;
		}
	} else {
		if (fitstable_close(uniform)) {
			ERROR("Failed to close output table");
			return -1;
		}
	}
	fitstable_close(catalog);

	// startree
	if (p->inmemory) {
		startag = fitstable_open_in_memory();

	} else {
		skdtfn = create_temp_file("skdt", p->tempdir);
		sl_append_nocopy(tempfiles, skdtfn);

		logverb("Reading uniformized catalog %s...\n", unifn);
		uniform = fitstable_open(unifn);
		if (!uniform) {
			ERROR("Failed to open uniformized catalog");
			return -1;
		}
	}

	{
		int Nleaf = 25;
		int datatype = KDT_DATA_U32;
		int treetype = KDT_TREE_U32;
		int buildopts = KD_BUILD_SPLIT;

		logverb("Building star kdtree from %i stars\n", fitstable_nrows(uniform));
		starkd = startree_build(uniform, p->racol, p->deccol, datatype, treetype,
								buildopts, Nleaf, p->args, p->argc);
		if (!starkd) {
			ERROR("Failed to create star kdtree");
			return -1;
		}

		if (!p->inmemory) {
			logverb("Writing star kdtree to %s\n", skdtfn);
			if (startree_write_to_file(starkd, skdtfn)) {
				ERROR("Failed to write star kdtree");
				return -1;
			}
			startree_close(starkd);

			startag = fitstable_open_for_appending(skdtfn);
			if (!startag) {
				ERROR("Failed to re-open star kdtree file %s for appending", skdtfn);
				return -1;
			}
		}

		logverb("Adding star kdtree tag-along data...\n");
		if (startree_write_tagalong_table(uniform, startag, p->racol, p->deccol)) {
			ERROR("Failed to write tag-along table");
			return -1;
		}
		if (p->inmemory) {
			if (fitstable_switch_to_reading(startag)) {
				ERROR("Failed to switch star tag-along data to read-mode");
				return -1;
			}
			starkd->tagalong = startag;

		} else {
			if (fitstable_close(startag)) {
				ERROR("Failed to close star kdtree tag-along data");
				return -1;
			}
		}
	}
	fitstable_close(uniform);

	// hpquads

	if (p->inmemory) {
		codes = codefile_open_in_memory();
		quads = quadfile_open_in_memory();
		if (hpquads(starkd, codes, quads, p->Nside,
					p->qlo, p->qhi, p->dimquads, p->passes, p->Nreuse, p->Nloosen,
					p->indexid, p->scanoccupied, p->args, p->argc)) {
			ERROR("hpquads failed");
			return -1;
		}
		if (quadfile_switch_to_reading(quads)) {
			ERROR("Failed to switch quadfile to read-mode");
			return -1;
		}
		if (codefile_switch_to_reading(codes)) {
			ERROR("Failed to switch codefile to read-mode");
			return -1;
		}


	} else {
		quadfn = create_temp_file("quad", p->tempdir);
		sl_append_nocopy(tempfiles, quadfn);
		codefn = create_temp_file("code", p->tempdir);
		sl_append_nocopy(tempfiles, codefn);

		if (hpquads_files(skdtfn, codefn, quadfn, p->Nside,
						  p->qlo, p->qhi, p->dimquads, p->passes, p->Nreuse, p->Nloosen,
						  p->indexid, p->scanoccupied, p->args, p->argc)) {
			ERROR("hpquads failed");
			return -1;
		}

	}

	// codetree
	if (p->inmemory) {
		logmsg("Building code kdtree from %i codes\n", codes->numcodes);
		logmsg("dim: %i\n", codefile_dimcodes(codes));
		codekd = codetree_build(codes, 0, 0, 0, 0, p->args, p->argc);
		if (!codekd) {
			ERROR("Failed to build code kdtree");
			return -1;
		}
		if (codefile_close(codes)) {
			ERROR("Failed to close codefile");
			return -1;
		}

	} else {
		ckdtfn = create_temp_file("ckdt", p->tempdir);
		sl_append_nocopy(tempfiles, ckdtfn);

		if (codetree_files(codefn, ckdtfn, 0, 0, 0, 0, p->args, p->argc)) {
			ERROR("codetree failed");
			return -1;
		}
	}

	// unpermute-stars
	logmsg("Unpermute-stars...\n");
	if (p->inmemory) {
		quads2 = quadfile_open_in_memory();
		if (unpermute_stars(starkd, quads, &starkd2, quads2,
							TRUE, FALSE, p->args, p->argc)) {
			ERROR("Failed to unpermute-stars");
			return -1;
		}
		if (quadfile_close(quads)) {
			ERROR("Failed to close in-memory quads");
			return -1;
		}
		if (quadfile_switch_to_reading(quads2)) {
			ERROR("Failed to switch quads2 to read-mode");
			return -1;
		}
		startag2 = fitstable_open_in_memory();
		startag2->table = fits_copy_table(startag->table);
		startag2->table->nr = 0;
		startag2->header = qfits_header_copy(startag->header);
		if (unpermute_stars_tagalong(starkd, startag2)) {
			ERROR("Failed to unpermute-stars tag-along data");
			return -1;
		}
		starkd2->tagalong = startag2;

		// unpermute-stars makes a shallow copy of the tree, so don't just startree_close(starkd)...
		free(starkd->tree->perm);
		free(starkd->tree);
		starkd->tree = NULL;
		startree_close(starkd);

	} else {
		skdt2fn = create_temp_file("skdt2", p->tempdir);
		sl_append_nocopy(tempfiles, skdt2fn);
		quad2fn = create_temp_file("quad2", p->tempdir);
		sl_append_nocopy(tempfiles, quad2fn);

		logmsg("Unpermuting stars from %s and %s to %s and %s\n", skdtfn, quadfn, skdt2fn, quad2fn);
		if (unpermute_stars_files(skdtfn, quadfn, skdt2fn, quad2fn,
								  TRUE, FALSE, p->args, p->argc)) {
			ERROR("Failed to unpermute-stars");
			return -1;
		}
	}


	// unpermute-quads
	logmsg("Unpermute-quads...\n");
	if (p->inmemory) {
		quads3 = quadfile_open_in_memory();
		if (unpermute_quads(quads2, codekd, quads3, &codekd2, p->args, p->argc)) {
			ERROR("Failed to unpermute-quads");
			return -1;
		}
		// unpermute-quads makes a shallow copy of the tree, so don't just codetree_close(codekd)...
		free(codekd->tree->perm);
		free(codekd->tree);
		codekd->tree = NULL;
		codetree_close(codekd);

		if (quadfile_switch_to_reading(quads3)) {
			ERROR("Failed to switch quads3 to read-mode");
			return -1;
		}
		if (quadfile_close(quads2)) {
			ERROR("Failed to close quadfile quads2");
			return -1;
		}

	} else {
		ckdt2fn = create_temp_file("ckdt2", p->tempdir);
		sl_append_nocopy(tempfiles, ckdt2fn);
		quad3fn = create_temp_file("quad3", p->tempdir);
		sl_append_nocopy(tempfiles, quad3fn);
		logmsg("Unpermuting quads from %s and %s to %s and %s\n", quad2fn, ckdtfn, quad3fn, ckdt2fn);
		if (unpermute_quads_files(quad2fn, ckdtfn,
								  quad3fn, ckdt2fn, p->args, p->argc)) {
			ERROR("Failed to unpermute-quads");
			return -1;
		}
	}

	// index
	if (p->inmemory) {
		index = index_build_from(codekd2, quads3, starkd2);
		if (!index) {
			ERROR("Failed to create index from constituent parts");
			return -1;
		}
		/* When closing:
		 kdtree_free(codekd2->tree);
		 codekd2->tree = NULL;
		 */
		*p_index = index;

	} else {
		logmsg("Merging %s and %s and %s to %s\n", quad3fn, ckdt2fn, skdt2fn, indexfn);
		if (merge_index_files(quad3fn, ckdt2fn, skdt2fn, indexfn)) {
			ERROR("Failed to merge-index");
			return -1;
		}
	}

	if (p->delete_tempfiles) {
		int i;
		for (i=0; i<sl_size(tempfiles); i++) {
			char* fn = sl_get(tempfiles, i);
			logverb("Deleting temp file %s\n", fn);
			if (unlink(fn))
				SYSERROR("Failed to delete temp file \"%s\"", fn);
		}
	}

	sl_free2(tempfiles);
	return 0;
}


int build_index_files(const char* infn, const char* indexfn,
					  index_params_t* p) {
	fitstable_t* catalog;

	logmsg("Reading %s...\n", infn);
	catalog = fitstable_open(infn);
    if (!catalog) {
        ERROR("Couldn't read catalog %s", infn);
		return -1;
    }
	logmsg("Got %i stars\n", fitstable_nrows(catalog));

	if (p->inmemory) {
		index_t* index;
		if (build_index(catalog, p, &index, NULL)) {
			return -1;
		}
		logmsg("Writing to file %s\n", indexfn);
		if (merge_index(index->quads, index->codekd, index->starkd, indexfn)) {
			ERROR("Failed to write index file");
			return -1;
		}
		kdtree_free(index->codekd->tree);
		index->codekd->tree = NULL;
		index_close(index);

	} else {
		if (build_index(catalog, p, NULL, indexfn)) {
			return -1;
		}
	}

	return 0;
}

void build_index_defaults(index_params_t* p) {
	memset(p, 0, sizeof(index_params_t));
	p->sweeps = 10;
	p->racol = "RA";
	p->deccol = "DEC";
	p->passes = 4;
	p->Nreuse = 2;
	p->dimquads = 4;
	p->sortasc = TRUE;
	// default to all-sky
	p->bighp = -1;
	//p->inmemory = TRUE;
	p->delete_tempfiles = TRUE;
	p->tempdir = "/tmp";
}

