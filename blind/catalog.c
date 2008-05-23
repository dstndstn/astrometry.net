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

#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "catalog.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "mathutil.h"

#define CHUNK_XYZ 0
#define CHUNK_MAG 1

static FILE* getfid(catalog* cat) {
    return fitsbin_get_fid(cat->fb);
}

static int callback_read_header(qfits_header* primheader, qfits_header* header,
								size_t* expected, char** errstr, void* userdata) {
	catalog* cat = userdata;

	cat->numstars = qfits_header_getint(primheader, "NSTARS", -1);
	cat->healpix = qfits_header_getint(primheader, "HEALPIX", -1);
	cat->hpnside = qfits_header_getint(primheader, "HPNSIDE", 1);
	if (fits_check_endian(primheader)) {
		if (errstr) *errstr = "Catalog file was written with wrong endianness.\n";
        return -1;
	}
	if (cat->numstars == -1) {
		if (errstr) *errstr = "Couldn't find NSTARS header in catalog file.";
        return -1;
	}

    *expected = cat->numstars * DIM_STARS * sizeof(double);
	return 0;
}

static int callback_read_mags(qfits_header* primheader, qfits_header* header,
                              size_t* expected, char** errstr, void* userdata) {
	catalog* cat = userdata;
    *expected = cat->numstars * sizeof(float);
	return 0;
}

static catalog* new_catalog() {
	catalog* cat;
	fitsbin_chunk_t* chunk;

    cat = calloc(1, sizeof(catalog));
	if (!cat) {
		fprintf(stderr, "catalog_open: malloc failed.\n");
	}
    cat->fb = fitsbin_new(2);

    // Star positions
    chunk = cat->fb->chunks + CHUNK_XYZ;
    chunk->tablename = strdup("xyz");
    chunk->required = 1;
    chunk->callback_read_header = callback_read_header;
    chunk->userdata = cat;
	chunk->itemsize = DIM_STARS * sizeof(double);

    // Star magnitudes
    chunk = cat->fb->chunks + CHUNK_MAG;
    chunk->tablename = strdup("mags");
    chunk->required = 0;
    chunk->callback_read_header = callback_read_mags;
    chunk->userdata = cat;
    chunk->itemsize = sizeof(float);

    return cat;
}

catalog* catalog_open(char* catfn) {
    catalog* cat = NULL;

    cat = new_catalog();
    if (!cat)
        goto bailout;

    fitsbin_set_filename(cat->fb, catfn);
    if (fitsbin_read(cat->fb)) {
        fprintf(stderr, "catalog: fitsbin_read() failed.\n");
		goto bailout;
	}
	cat->stars = cat->fb->chunks[CHUNK_XYZ].data;
    cat->mags  = cat->fb->chunks[CHUNK_MAG].data;
    return cat;

 bailout:
    if (cat)
        catalog_close(cat);
    return NULL;
}

static void add_default_header_vals(catalog* cat) {
	qfits_header* hdr;
	hdr = catalog_get_header(cat);
	qfits_header_add(hdr, "AN_FILE", "OBJS", "This file has a list of object positions.", NULL);
    fits_add_endian(hdr);
	fits_add_double_size(hdr);
	qfits_header_add(hdr, "NSTARS", "0", "Number of stars in this file.", NULL);
	qfits_header_add(hdr, "HEALPIX", "-1", "Healpix covered by this catalog, with Nside=HPNSIDE", NULL);
	qfits_header_add(hdr, "HPNSIDE", "-1", "Nside of HEALPIX.", NULL);
	qfits_header_add(hdr, "COMMENT", "This is a flat array of XYZ for each catalog star.", NULL, NULL);
	qfits_header_add(hdr, "COMMENT", "  (ie, star position on the unit sphere)", NULL, NULL);
	qfits_header_add(hdr, "COMMENT", "  (stored as three native-{endian,size} doubles)", NULL, NULL);
}

catalog* catalog_open_for_writing(char* fn)  {
	catalog* cat;

	cat = new_catalog();
	if (!cat)
		goto bailout;

    fitsbin_set_filename(cat->fb, fn);
    if (fitsbin_start_write(cat->fb))
		//fprintf(stderr, "%s\n", errstr);
        goto bailout;

    cat->hpnside = 1;

    add_default_header_vals(cat);
	return cat;

 bailout:
	if (cat)
        catalog_close(cat);
	return NULL;
}

int catalog_write_to_file(catalog* cat, char* fn) {
    FILE* fid;
    fitsbin_set_filename(cat->fb, fn);
    if (fitsbin_start_write(cat->fb)) {
        fprintf(stderr, "Failed to write catalog.\n");
        return -1;
    }
    add_default_header_vals(cat);

    if (catalog_write_header(cat) ||
        catalog_fix_header(cat)) {
        fprintf(stderr, "Failed to write header.\n");
        return -1;
    }

    if (fitsbin_write_items(cat->fb, CHUNK_XYZ, cat->stars, cat->numstars)) {
		fprintf(stderr, "Failed to write catalog star to file %s: %s.\n",
		        fn, strerror(errno));
        return -1;
    }

    fid = getfid(cat);
    fits_pad_file(fid);

	if (fclose(fid)) {
		fflush(stdout);
		fprintf(stderr, "Couldn't close catalog file %s: %s\n",
		        fn, strerror(errno));
		return -1;
	}
	return 0;
}

qfits_header* catalog_get_header(catalog* cat) {
    return fitsbin_get_primary_header(cat->fb);
}

int catalog_write_header(catalog* cat) {
	fitsbin_t* fb = cat->fb;
	fb->chunks[CHUNK_XYZ].nrows = cat->numstars;

	if (fitsbin_write_primary_header(fb) ||
		fitsbin_write_header(fb)) {
		fprintf(stderr, "Failed to write catalog header.\n");
		return -1;
	}
	return 0;
}

int catalog_fix_header(catalog* cat) {
	qfits_header* hdr;
	fitsbin_t* fb = cat->fb;
	fb->chunks[CHUNK_XYZ].nrows = cat->numstars;

	hdr = catalog_get_header(cat);
	// fill in the real values...
    fits_header_mod_int(hdr, "NSTARS", cat->numstars, "Number of stars.");
    fits_header_mod_int(hdr, "HEALPIX", cat->healpix, "Healpix covered by this catalog, with Nside=HPNSIDE");
    fits_header_mod_int(hdr, "HPNSIDE", cat->hpnside, "Nside of HEALPIX.");

	if (fitsbin_fix_primary_header(fb) ||
		fitsbin_fix_header(fb)) {
        fprintf(stderr, "Failed to fix catalog header.\n");
		return -1;
	}
	return 0;
}

void catalog_compute_radecminmax(catalog* cat) {
	double ramin, ramax, decmin, decmax;
	int i;
	ramin = HUGE_VAL;
	ramax = -HUGE_VAL;
	decmin = HUGE_VAL;
	decmax = -HUGE_VAL;
	for (i = 0; i < cat->numstars; i++) {
		double* xyz;
		double ra, dec;
		xyz = catalog_get_star(cat, i);
		ra = xy2ra(xyz[0], xyz[1]);
		dec = z2dec(xyz[2]);
		if (ra > ramax)
			ramax = ra;
		if (ra < ramin)
			ramin = ra;
		if (dec > decmax)
			decmax = dec;
		if (dec < decmin)
			decmin = dec;
	}
	cat->ramin = ramin;
	cat->ramax = ramax;
	cat->decmin = decmin;
	cat->decmax = decmax;
}

double* catalog_get_base(catalog* cat) {
	return cat->stars;
}

double* catalog_get_star(catalog* cat, int sid) {
	if (sid >= cat->numstars) {
		fflush(stdout);
		fprintf(stderr, "catalog: asked for star %u, but catalog size is only %u.\n",
		        sid, cat->numstars);
		return NULL;
	}
	return cat->stars + sid * 3;
}

int catalog_write_star(catalog* cat, double* star) {
	if (fitsbin_write_item(cat->fb, CHUNK_XYZ, star)) {
		fprintf(stderr, "Failed to write catalog data to file %s: %s\n",
                cat->fb->filename, strerror(errno));
		return -1;
	}
	cat->numstars++;
	return 0;
}

int catalog_write_mags(catalog* cat) {
	fitsbin_chunk_t* chunk;

	if (fits_pad_file(getfid(cat))) {
		fflush(stdout);
		fprintf(stderr, "Failed to pad catalog FITS file.\n");
		return -1;
	}

    chunk = cat->fb->chunks + CHUNK_MAG;
    chunk->nrows = cat->numstars;

    if (fitsbin_write_chunk_header(cat->fb, CHUNK_MAG)) {
        fprintf(stderr, "Failed to write magnitudes header.\n");
        return -1;
    }

	if (fitsbin_write_items(cat->fb, CHUNK_MAG, cat->mags, cat->numstars)) {
		fprintf(stderr, "Failed to write catalog magnitudes: %s.\n",
		        strerror(errno));
		return -1;
	}

	if (fits_pad_file(getfid(cat))) {
		fflush(stdout);
		fprintf(stderr, "Failed to pad catalog FITS file.\n");
		return -1;
	}
	return 0;
}

int catalog_close(catalog* cat) {
    int rtn;
	if (!cat) return 0;
	rtn = fitsbin_close(cat->fb);
	free(cat);
    return rtn;
}

