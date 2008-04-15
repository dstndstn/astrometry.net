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
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>

#include "qfits.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "permutedsort.h"

static const char* OPTIONS = "hc:i:o:dq";

static void printHelp(char* progname) {
	printf("%s    -i <input-file>\n"
		   "      -o <output-file>\n"
		   "      -c <column-name>\n"
		   "      [-d]: sort in descending order (default, ascending)\n",
		   progname);
}

static int sort_doubles_desc(const void* v1, const void* v2) {
	double d1 = *((double*)v1);
	double d2 = *((double*)v2);
	if (d1 > d2)
		return -1;
	if (d1 == d2)
		return 0;
	return 1;
}

static int sort_doubles_asc(const void* v1, const void* v2) {
	double d1 = *((double*)v1);
	double d2 = *((double*)v2);
	if (d1 < d2)
		return -1;
	if (d1 == d2)
		return 0;
	return 1;
}

static int sort_floats_desc(const void* v1, const void* v2) {
	float d1 = *((float*)v1);
	float d2 = *((float*)v2);
	if (d1 > d2)
		return -1;
	if (d1 == d2)
		return 0;
	return 1;
}

static int sort_floats_asc(const void* v1, const void* v2) {
	float d1 = *((float*)v1);
	float d2 = *((float*)v2);
	if (d1 < d2)
		return -1;
	if (d1 == d2)
		return 0;
	return 1;
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	FILE* fin = NULL;
	FILE* fout = NULL;
	char* colname = NULL;
	char* progname = argv[0];
	int nextens;
	int ext;
	int start, size;
	int descending = 0;
	unsigned char* buffer;
    bool verbose = TRUE;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'q':
            verbose = FALSE;
            break;
        case 'c':
			colname = optarg;
            break;
        case 'i':
			infn = optarg;
			break;
        case 'o':
			outfn = optarg;
			break;
		case 'd':
			descending = 1;
			break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!infn || !outfn || !colname) {
		printHelp(progname);
		exit(-1);
	}

	if (infn) {
		fin = fopen(infn, "rb");
		if (!fin) {
			fprintf(stderr, "Failed to open input file %s: %s\n", infn, strerror(errno));
			exit(-1);
		}
	}

	if (outfn) {
		fout = fopen(outfn, "wb");
		if (!fout) {
			fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
			exit(-1);
		}
	}

	// copy the main header exactly.
	if (qfits_get_hdrinfo(infn, 0, &start, &size)) {
		fprintf(stderr, "Couldn't get main header.\n");
		exit(-1);
	}
	buffer = malloc(size);
	if (fread(buffer, 1, size, fin) != size) {
		fprintf(stderr, "Error reading main header: %s\n", strerror(errno));
		exit(-1);
	}
	if (fwrite(buffer, 1, size, fout) != size) {
		fprintf(stderr, "Error writing main header: %s\n", strerror(errno));
		exit(-1);
	}
	free(buffer);

	nextens = qfits_query_n_ext(infn);
    if (verbose)
        printf("Sorting %i extensions.\n", nextens);
	buffer = NULL;
	for (ext=1; ext<=nextens; ext++) {
		int c;
		qfits_table* table;
		qfits_col* col;
		unsigned char* map;
		size_t mapsize;
		int mgap;
		off_t mstart;
		size_t msize;
		int atomsize;
		int (*sort_func)(const void*, const void*);
		int* perm;
		unsigned char* tabledata;
		unsigned char* tablehdr;
		int hdrstart, hdrsize, datsize, datstart;
		int i;

		if (!qfits_is_table(infn, ext)) {
            fprintf(stderr, "Extention %i isn't a table. Skipping.\n", ext);
			continue;
		}
		table = qfits_table_open(infn, ext);
		if (!table) {
			fprintf(stderr, "failed to open table: file %s, extension %i. Skipping.\n", infn, ext);
			continue;
		}
		c = fits_find_column(table, colname);
		if (c == -1) {
			fprintf(stderr, "Couldn't find column named \"%s\" in extension %i.  Skipping.\n", colname, ext);
			continue;
		}
		col = table->col + c;
		switch (col->atom_type) {
		case TFITS_BIN_TYPE_D:
			buffer = realloc(buffer, table->nr * sizeof(double));
			if (descending)
				sort_func = sort_doubles_desc;
			else
				sort_func = sort_doubles_asc;
			break;
		case TFITS_BIN_TYPE_E:
			buffer = realloc(buffer, table->nr * sizeof(float));
			if (descending)
				sort_func = sort_floats_desc;
			else
				sort_func = sort_floats_asc;
			break;
		default:
			fprintf(stderr, "Column %s is neither FITS type D nor E.  Skipping.\n", colname);
			continue;
		}
		atomsize = fits_get_atom_size(col->atom_type);

		qfits_query_column_seq_to_array(table, c, 0, table->nr,
										buffer, atomsize);

		perm = permuted_sort(buffer, atomsize, sort_func, NULL, table->nr);

		if (qfits_get_hdrinfo(infn, ext, &hdrstart, &hdrsize) ||
			qfits_get_datinfo(infn, ext, &datstart, &datsize)) {
			fprintf(stderr, "Couldn't get extension %i header or data extent.\n", ext);
			exit(-1);
		}
		start = hdrstart;
		size = hdrsize + datsize;
		get_mmap_size(start, size, &mstart, &msize, &mgap);

		mapsize = msize;
		map = mmap(NULL, mapsize, PROT_READ, MAP_SHARED, fileno(fin), mstart);
		if (map == MAP_FAILED) {
			fprintf(stderr, "Failed to mmap input file: %s\n", strerror(errno));
			exit(-1);
		}
		tabledata = map + mgap + (datstart - hdrstart);
		tablehdr  = map + mgap;

		if (fwrite(tablehdr, 1, hdrsize, fout) != hdrsize) {
			fprintf(stderr, "Failed to write table header: %s\n", strerror(errno));
			exit(-1);
		}

		for (i=0; i<table->nr; i++) {
			unsigned char* rowptr;
			rowptr = tabledata + perm[i] * table->tab_w;
			if (fwrite(rowptr, 1, table->tab_w, fout) != table->tab_w) {
				fprintf(stderr, "Failed to write: %s\n", strerror(errno));
				exit(-1);
			}
		}

		munmap(map, mapsize);
		free(perm);

		if (fits_pad_file(fout)) {
			fprintf(stderr, "Failed to add padding to extension %i.\n", ext);
			exit(-1);
		}

        qfits_table_close(table);
	}
	free(buffer);

	if (fclose(fout)) {
		fprintf(stderr, "Error closing output file: %s\n", strerror(errno));
	}
	fclose(fin);

	return 0;
}
