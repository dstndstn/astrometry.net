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

#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>

#include "qfits.h"
#include "fitsioutils.h"
#include "idfile.h"
#include "starutil.h"
#include "ioutils.h"

#define CHUNK_IDS 0

static int callback_read_header(qfits_header* primheader, qfits_header* header,
								size_t* expected, char** errstr,
								void* userdata) {
	idfile* id = userdata;

    id->numstars = qfits_header_getint(primheader, "NSTARS", -1);
	//qf->indexid = qfits_header_getint(primheader, "INDEXID", 0);
	id->healpix = qfits_header_getint(primheader, "HEALPIX", -1);

	if (id->numstars == -1) {
		if (errstr) *errstr = "Couldn't find NSTARS entries in FITS header.";
		return -1;
	}
    if (fits_check_endian(primheader)) {
		if (errstr) *errstr = "File was written with the wrong endianness.";
		return -1;
    }

    id->fb->chunks[CHUNK_IDS].itemsize = sizeof(uint64_t);

    *expected = id->numstars * sizeof(uint64_t);
	return 0;
}

static idfile* new_idfile() {
	idfile* id = calloc(1, sizeof(idfile));
	fitsbin_chunk_t* chunk;

	if (!id) {
		fflush(stdout);
		fprintf(stderr, "Couldn't malloc a idfile struct: %s\n", strerror(errno));
		return NULL;
	}
	id->healpix = -1;

    id->fb = fitsbin_new(1);

    chunk = id->fb->chunks + CHUNK_IDS;
    chunk->tablename = strdup("ids");
    chunk->required = 1;
    chunk->callback_read_header = callback_read_header;
    chunk->userdata = id;

	return id;
}

idfile* idfile_open(char* fn) {
    idfile* id = NULL;

	id = new_idfile();
	if (!id)
		goto bailout;

    fitsbin_set_filename(id->fb, fn);
    if (fitsbin_read(id->fb))
        goto bailout;

	id->anidarray = id->fb->chunks[CHUNK_IDS].data;
	return id;

bailout:
    if (id)
        idfile_close(id);
	return NULL;
}

int idfile_close(idfile* id) {
    int rtn;
	if (!id) return 0;
	rtn = fitsbin_close(id->fb);
	free(id);
    return rtn;
}

idfile* idfile_open_for_writing(char* fn) {
	idfile* id;
	qfits_header* hdr;

	id = new_idfile();
	if (!id)
		goto bailout;

    fitsbin_set_filename(id->fb, fn);
    if (fitsbin_start_write(id->fb))
        goto bailout;

	// the header
    hdr = idfile_get_header(id);
    fits_add_endian(hdr);

	// These are be placeholder values...
	qfits_header_add(hdr, "AN_FILE", "ID", "This file lists Astrometry.net star IDs for catalog stars.", NULL);
	qfits_header_add(hdr, "NSTARS", "0", "Number of stars used.", NULL);
	qfits_header_add(hdr, "HEALPIX", "-1", "Healpix covered by this file.", NULL);
	qfits_header_add(hdr, "COMMENT", "This is a flat array of ANIDs for each catalog star.", NULL, NULL);
	qfits_header_add(hdr, "COMMENT", " (each A.N id is a native-endian uint64)", NULL, NULL);

	return id;

bailout:
	if (id)
        idfile_close(id);
	return NULL;
}

int idfile_write_anid(idfile* id, uint64_t anid) {
    if (fitsbin_write_item(id->fb, CHUNK_IDS, &anid)) {
        fprintf(stderr, "idfile_fits_write_anid: failed to write: %s\n", strerror(errno));
		return -1;
	}
    id->numstars++;
    return 0;
}

qfits_header* idfile_get_header(idfile* id) {
    return fitsbin_get_primary_header(id->fb);
}

int idfile_fix_header(idfile* id) {
	qfits_header* hdr;
	fitsbin_t* fb = id->fb;

	fb->chunks[CHUNK_IDS].nrows = id->numstars;

	hdr = idfile_get_header(id);

	// fill in the real values...
	fits_header_mod_int(hdr, "NSTARS", id->numstars, "Number of stars.");
	fits_header_mod_int(hdr, "HEALPIX", id->healpix, "Healpix covered by this file.");

	if (fitsbin_fix_primary_header(fb) ||
		fitsbin_fix_header(fb)) {
        fprintf(stderr, "Failed to fix idfile header.\n");
		return -1;
	}
	return 0;
}

int idfile_write_header(idfile* id) {
	fitsbin_t* fb = id->fb;
	fb->chunks[CHUNK_IDS].nrows = id->numstars;

	if (fitsbin_write_primary_header(fb) ||
		fitsbin_write_header(fb)) {
		fprintf(stderr, "Failed to write quadfile header.\n");
		return -1;
	}
	return 0;
}

uint64_t idfile_get_anid(idfile* id, uint starid) {
	if (starid >= id->numstars) {
		fflush(stdout);
		fprintf(stderr, "Requested quadid %i, but number of quads is %i. SKY IS FALLING\n",
		        starid, id->numstars);
		assert(0);
		return *(int*)0x0; /* explode gracefully */
	}
	return id->anidarray[starid];
}
