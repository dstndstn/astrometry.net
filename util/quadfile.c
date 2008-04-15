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
#include <assert.h>

#include "qfits.h"
#include "fitsioutils.h"
#include "quadfile.h"
#include "starutil.h"
#include "ioutils.h"

#define CHUNK_QUADS 0

static int callback_read_header(qfits_header* primheader, qfits_header* header,
								size_t* expected, char** errstr,
								void* userdata) {
	quadfile* qf = userdata;

    qf->dimquads = qfits_header_getint(primheader, "DIMQUADS", 4);
    qf->numquads = qfits_header_getint(primheader, "NQUADS", -1);
    qf->numstars = qfits_header_getint(primheader, "NSTARS", -1);
    qf->index_scale_upper = qfits_header_getdouble(primheader, "SCALE_U", -1.0);
    qf->index_scale_lower = qfits_header_getdouble(primheader, "SCALE_L", -1.0);
	qf->indexid = qfits_header_getint(primheader, "INDEXID", 0);
	qf->healpix = qfits_header_getint(primheader, "HEALPIX", -1);

	if ((qf->numquads == -1) || (qf->numstars == -1) ||
		(qf->index_scale_upper == -1.0) || (qf->index_scale_lower == -1.0)) {
        asprintf(errstr, "Couldn't find NQUADS or NSTARS or SCALE_U or SCALE_L entries in FITS header.");
		return -1;
	}
    if (fits_check_endian(primheader)) {
        asprintf(errstr, "File was written with the wrong endianness.");
		return -1;
    }

    qf->fb->chunks[CHUNK_QUADS].itemsize = qf->dimquads * sizeof(uint32_t);

    *expected = qf->numquads * qf->dimquads * sizeof(uint32_t);
	return 0;
}

static quadfile* new_quadfile() {
	quadfile* qf = calloc(1, sizeof(quadfile));
	fitsbin_chunk_t* chunk;

	if (!qf) {
		fprintf(stderr, "Couldn't malloc a quadfile struct: %s\n", strerror(errno));
		return NULL;
	}
	qf->healpix = -1;

    qf->fb = fitsbin_new(1);

    chunk = qf->fb->chunks + CHUNK_QUADS;
    chunk->tablename = strdup("quads");
    chunk->required = 1;
    chunk->callback_read_header = callback_read_header;
    chunk->userdata = qf;
    
	return qf;
}

int quadfile_dimquads(const quadfile* qf) {
    return qf->dimquads;
}

qfits_header* quadfile_get_header(const quadfile* qf) {
	return fitsbin_get_primary_header(qf->fb);
}

quadfile* quadfile_open(const char* fn) {
    quadfile* qf = NULL;

    qf = new_quadfile();
    if (!qf)
        goto bailout;

    fitsbin_set_filename(qf->fb, fn);
    if (fitsbin_read(qf->fb)) {
        fprintf(stderr, "quadfile_open: fitsbin_read failed.\n");
        goto bailout;
    }
	qf->quadarray = qf->fb->chunks[CHUNK_QUADS].data;
    return qf;

 bailout:
    if (qf)
        quadfile_close(qf);
    return NULL;
}

int quadfile_close(quadfile* qf) {
    int rtn;
	if (!qf) return 0;
	rtn = fitsbin_close(qf->fb);
	free(qf);
    return rtn;
}

quadfile* quadfile_open_for_writing(const char* fn) {
	quadfile* qf;
	qfits_header* hdr;

	qf = new_quadfile();
	if (!qf)
		goto bailout;

    qf->dimquads = 4;
    fitsbin_set_filename(qf->fb, fn);

    if (fitsbin_start_write(qf->fb))
        goto bailout;

	// add default values to header
	hdr = fitsbin_get_primary_header(qf->fb);
    fits_add_endian(hdr);
	qfits_header_add(hdr, "AN_FILE", "QUAD", "This file lists, for each quad, its stars.", NULL);
	qfits_header_add(hdr, "DIMQUADS", "0", "Number of stars in a quad.", NULL);
	qfits_header_add(hdr, "NQUADS", "0", "Number of quads.", NULL);
	qfits_header_add(hdr, "NSTARS", "0", "Number of stars used (or zero).", NULL);
	qfits_header_add(hdr, "SCALE_U", "0.0", "Upper-bound index scale.", NULL);
	qfits_header_add(hdr, "SCALE_L", "0.0", "Lower-bound index scale.", NULL);
	qfits_header_add(hdr, "INDEXID", "0", "Index unique ID.", NULL);
	qfits_header_add(hdr, "HEALPIX", "-1", "Healpix of this index.", NULL);
    fits_add_long_comment(hdr, "The first extension contains the quads "
                          "stored as %i 32-bit native-endian unsigned ints.", qf->dimquads);
	return qf;

 bailout:
	if (qf)
		quadfile_close(qf);
	return NULL;
}

int quadfile_write_header(quadfile* qf) {
	fitsbin_t* fb = qf->fb;
	fb->chunks[CHUNK_QUADS].itemsize = qf->dimquads * sizeof(uint32_t);
	fb->chunks[CHUNK_QUADS].nrows = qf->numquads;

	if (fitsbin_write_primary_header(fb) ||
		fitsbin_write_header(fb)) {
		fprintf(stderr, "Failed to write quadfile header.\n");
		return -1;
	}
	return 0;
}

int quadfile_write_quad(quadfile* qf, uint* stars) {
	uint32_t* data;
	uint32_t ustars[qf->dimquads];
	int i;
	if (sizeof(uint32_t) == sizeof(uint)) {
		data = stars;
	} else {
		data = ustars;
		for (i=0; i<qf->dimquads; i++)
			ustars[i] = stars[i];
	}
    if (fitsbin_write_item(qf->fb, CHUNK_QUADS, data)) {
		fprintf(stderr, "quadfile_write_quad: failed to write: %s\n", strerror(errno));
		return -1;
	}
	qf->numquads++;
	return 0;
}

int quadfile_fix_header(quadfile* qf) {
	qfits_header* hdr;
	fitsbin_t* fb = qf->fb;

	fb->chunks[CHUNK_QUADS].itemsize = qf->dimquads * sizeof(uint32_t);
	fb->chunks[CHUNK_QUADS].nrows = qf->numquads;

	hdr = fitsbin_get_primary_header(fb);

	// fill in the real values...
	fits_header_mod_int(hdr, "DIMQUADS", qf->dimquads, "Number of stars in a quad.");
	fits_header_mod_int(hdr, "NQUADS", qf->numquads, "Number of quads.");
	fits_header_mod_int(hdr, "NSTARS", qf->numstars, "Number of stars.");
	fits_header_mod_double(hdr, "SCALE_U", qf->index_scale_upper, "Upper-bound index scale (radians).");
	fits_header_mod_double(hdr, "SCALE_L", qf->index_scale_lower, "Lower-bound index scale (radians).");
	fits_header_mod_int(hdr, "INDEXID", qf->indexid, "Index unique ID.");
	fits_header_mod_int(hdr, "HEALPIX", qf->healpix, "Healpix of this index.");

	if (fitsbin_fix_primary_header(fb) ||
		fitsbin_fix_header(fb)) {
        fprintf(stderr, "Failed to fix quad header.\n");
		return -1;
	}
	return 0;
}

double quadfile_get_index_scale_upper_arcsec(const quadfile* qf) {
    return rad2arcsec(qf->index_scale_upper);
}

double quadfile_get_index_scale_lower_arcsec(const quadfile* qf) {
	return rad2arcsec(qf->index_scale_lower);
}

int quadfile_get_stars(const quadfile* qf, uint quadid, uint* stars) {
    int i;
	if (quadid >= qf->numquads) {
		fprintf(stderr, "Requested quadid %i, but number of quads is %i.\n",
				quadid, qf->numquads);
        assert(quadid < qf->numquads);
		return -1;
	}

    for (i=0; i<qf->dimquads; i++) {
        stars[i] = qf->quadarray[quadid * qf->dimquads + i];
    }
    return 0;
}
