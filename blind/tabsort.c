/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.
  Copyright 2010 Dustin Lang.

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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "qfits.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "errors.h"

int tabsort(const char* infn, const char* outfn, const char* colname,
            anbool descending) {
	FILE* fin;
	FILE* fout;
	int ext, nextens;
	int start, size;
    void* data = NULL;
    int* perm = NULL;
    unsigned char* map = NULL;
    size_t mapsize = 0;

    fin = fopen(infn, "rb");
    if (!fin) {
        SYSERROR("Failed to open input file %s", infn);
        return -1;
    }

    fout = fopen(outfn, "wb");
    if (!fout) {
        SYSERROR("Failed to open output file %s", outfn);
        goto bailout;
    }

	// copy the main header exactly.
	if (qfits_get_hdrinfo(infn, 0, &start, &size)) {
		ERROR("Failed to read primary FITS header.");
        goto bailout;
	}

    if (pipe_file_offset(fin, start, size, fout)) {
        ERROR("Failed to copy primary FITS header.");
        goto bailout;
    }

	nextens = qfits_query_n_ext(infn);
    //logverb("Sorting %i extensions.\n", nextens);
	for (ext=1; ext<=nextens; ext++) {
		int c;
		qfits_table* table;
		qfits_col* col;
		int mgap;
		off_t mstart;
		size_t msize;
		int atomsize;
		int (*sort_func)(const void*, const void*);
		unsigned char* tabledata;
		unsigned char* tablehdr;
		int hdrstart, hdrsize, datsize, datstart;
		int i;

		if (qfits_get_hdrinfo(infn, ext, &hdrstart, &hdrsize) ||
			qfits_get_datinfo(infn, ext, &datstart, &datsize)) {
			ERROR("Couldn't get extension %i header or data extent.", ext);
            goto bailout;
        }
		if (!qfits_is_table(infn, ext)) {
            ERROR("Extention %i isn't a table. Skipping.\n", ext);
			continue;
		}
		table = qfits_table_open(infn, ext);
		if (!table) {
			ERROR("Failed to open table: file %s, extension %i. Skipping.", infn, ext);
			continue;
		}
		c = fits_find_column(table, colname);
		if (c == -1) {
			ERROR("Couldn't find column named \"%s\" in extension %i.  Skipping.", colname, ext);
			continue;
		}
		col = table->col + c;
		switch (col->atom_type) {
		case TFITS_BIN_TYPE_D:
			data = realloc(data, table->nr * sizeof(double));
			if (descending)
				sort_func = compare_doubles_desc;
			else
				sort_func = compare_doubles_asc;
			break;
		case TFITS_BIN_TYPE_E:
			data = realloc(data, table->nr * sizeof(float));
			if (descending)
				sort_func = compare_floats_desc;
			else
				sort_func = compare_floats_asc;
			break;
		default:
			ERROR("Column %s is neither FITS type D nor E.  Skipping.", colname);
			continue;
		}

        // Grab the sort column.
		atomsize = fits_get_atom_size(col->atom_type);
		qfits_query_column_seq_to_array(table, c, 0, table->nr, data, atomsize);
        // Sort it.
		perm = permuted_sort(data, atomsize, sort_func, NULL, table->nr);

        // mmap the input file.
		start = hdrstart;
		size = hdrsize + datsize;
		get_mmap_size(start, size, &mstart, &msize, &mgap);
		mapsize = msize;
		map = mmap(NULL, mapsize, PROT_READ, MAP_SHARED, fileno(fin), mstart);
		if (map == MAP_FAILED) {
			SYSERROR("Failed to mmap input file %s", infn);
            map = NULL;
            goto bailout;
		}
		tabledata = map + mgap + (datstart - hdrstart);
		tablehdr  = map + mgap;

        // Copy the table header without change.
		if (fwrite(tablehdr, 1, hdrsize, fout) != hdrsize) {
			SYSERROR("Failed to write FITS table header");
            goto bailout;
		}

		for (i=0; i<table->nr; i++) {
			unsigned char* rowptr;
			rowptr = tabledata + perm[i] * table->tab_w;
			if (fwrite(rowptr, 1, table->tab_w, fout) != table->tab_w) {
				SYSERROR("Failed to write FITS table row");
                goto bailout;
			}
		}

		munmap(map, mapsize);
        map = NULL;
		free(perm);
        perm = NULL;

		if (fits_pad_file(fout)) {
			ERROR("Failed to add padding to extension %i", ext);
            goto bailout;
		}

        qfits_table_close(table);
	}
	free(data);

	if (fclose(fout)) {
		SYSERROR("Error closing output file");
        fout = NULL;
        goto bailout;
	}
	fclose(fin);
	return 0;

 bailout:
    free(data);
    free(perm);
    if (fout)
        fclose(fout);
    fclose(fin);
    if (map)
        munmap(map, mapsize);
    return -1;
}

