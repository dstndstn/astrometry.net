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

#include "qfits.h"
#include "bl.h"
#include "fitsioutils.h"
#include "ioutils.h"

char* OPTIONS = "hc:i:o:";

void printHelp(char* progname) {
	printf("%s    -i <input-file>\n"
		   "      -o <output-file>\n"
		   "      -c <column-name> ...\n\n",
		   progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char *argv[]) {
    int argchar;

	char* infn = NULL;
	char* outfn = NULL;
	FILE* fin = NULL;
	FILE* fout = NULL;
	pl* cols;
	char* progname = argv[0];
	int nextens;
	int ext;
	int NC;
	int start, size;

	qfits_table* outtable;
	unsigned char* buffer;

	cols = pl_new(16);

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'c':
			pl_append(cols, optarg);
            break;
        case 'i':
			infn = optarg;
			break;
        case 'o':
			outfn = optarg;
			break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

	if (!infn || !outfn || !pl_size(cols)) {
		printHelp(progname);
		exit(-1);
	}

    fin = fopen(infn, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input file %s: %s\n", infn, strerror(errno));
        exit(-1);
    }

    fout = fopen(outfn, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
        exit(-1);
    }

	// copy the main header exactly.
	if (qfits_get_hdrinfo(infn, 0, &start, &size)) {
		fprintf(stderr, "Couldn't get main header.\n");
		exit(-1);
	}
    if (pipe_file_offset(fin, start, size, fout)) {
        fprintf(stderr, "Failed to copy primary header.\n");
        exit(-1);
    }

	NC = pl_size(cols);
	nextens = qfits_query_n_ext(infn);
	printf("Translating %i extensions.\n", nextens);
	buffer = NULL;
	for (ext=1; ext<=nextens; ext++) {
		int c2, c;
		int columns[NC];
		int sizes[NC];
		int offsets[NC];
		int offset = 0;
		int off, n;
		int totalsize = 0;
		const int BLOCK=1000;
		qfits_table* table;
		qfits_header* header;
		qfits_header* tablehdr;

		if (ext%100 == 0) {
			printf("Extension %i.\n", ext);
			fflush(stdout);
		}

		if (!qfits_is_table(infn, ext)) {
			fprintf(stderr, "extention %i isn't a table.\n", ext);
			// HACK - just copy it.
			return -1;
		}
		table = qfits_table_open(infn, ext);
		if (!table) {
			fprintf(stderr, "failed to open table: file %s, extension %i.\n", infn, ext);
			return -1;
		}
		header = qfits_header_readext(infn, ext);
		if (!header) {
			fprintf(stderr, "failed to read header: extension %i\n", ext);
			exit(-1);
		}

		outtable = qfits_table_new(outfn, QFITS_BINTABLE, 0, NC, table->nr);
		outtable->tab_w = 0;
		
		for (c=0; c<pl_size(cols); c++) {
			columns[c] = -1;
		}
		for (c=0; c<pl_size(cols); c++) {
			char* colname = pl_get(cols, c);
			qfits_col* col;
			c2 = fits_find_column(table, colname);
			if (c2 == -1) {
				fprintf(stderr, "Extension %i: failed to find column named %s\n",
						ext, colname);
				exit(-1);
			}
			col = table->col + c2;
			columns[c] = c2;
			sizes[c] = col->atom_nb * col->atom_size;
			offsets[c] = offset;
			offset += sizes[c];

			qfits_col_fill(outtable->col + c,
						   col->atom_nb, col->atom_dec_nb,
						   col->atom_size, col->atom_type,
						   col->tlabel, col->tunit,
						   col->nullval, col->tdisp,
						   col->zero_present,
						   col->zero,
						   col->scale_present,
						   col->scale,
						   outtable->tab_w);
			outtable->tab_w += sizes[c];
		}
		totalsize = offset;

		tablehdr = qfits_table_ext_header_default(outtable);
		// add any headers from the original table that aren't part of the BINTABLE extension.
        fits_copy_non_table_headers(tablehdr, header);
		qfits_header_dump(tablehdr, fout);
		qfits_header_destroy(tablehdr);
		
		buffer = realloc(buffer, totalsize * BLOCK);

		for (off=0; off<table->nr; off+=n) {
			if (off + BLOCK > table->nr)
				n = table->nr - off;
			else
				n = BLOCK;
			for (c=0; c<pl_size(cols); c++)
				qfits_query_column_seq_to_array_no_endian_swap
					(table, columns[c], off, n, buffer + offsets[c], totalsize);
			if (fwrite(buffer, totalsize, n, fout) != n) {
				fprintf(stderr, "Error writing a block of data: ext %i: %s\n", ext, strerror(errno));
				exit(-1);
			}
		}

		qfits_table_close(outtable);

		fits_pad_file(fout);

		qfits_header_destroy(header);
		qfits_table_close(table);
	}
	free(buffer);

	if (fclose(fout)) {
		fprintf(stderr, "Error closing output file: %s\n", strerror(errno));
	}
	fclose(fin);

	pl_free(cols);
	return 0;
}
