/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "fitstable.h"
#include "anqfits.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "errors.h"
#include "log.h"

int cut_table(const char* infn, const char* outfn, int N) {
    fitstable_t* in;
    //fitstable_t* out;
    FILE* fid;
    FILE* fin;
    int i, Next;

    in = fitstable_open(infn);
    if (!in) {
        ERROR("Failed to read input file %s", infn);
        return -1;
    }
    /*
     out = fitstable_open_for_writing(outfn);
     if (!out) {
     ERROR("Failed to out output file %s", outfn);
     return -1;
     }
     */
    fid = fopen(outfn, "wb");
    if (!fid) {
        ERROR("Failed to open output file %s", outfn);
        return -1;
    }

    fin = fopen(infn, "rb");
    if (!fin) {
        ERROR("Failed to open input file %s", infn);
        return -1;
    }

    /*
     fitstable_set_primary_header(out, fitstable_get_primary_header(in));
     if (fitstable_write_primary_header(out)) {
     ERROR("Failed to write primary header");
     return -1;
     }
     */

    if (qfits_header_dump(fitstable_get_primary_header(in), fid)) {
        ERROR("Failed to write primary header");
        return -1;
    }
	
    Next = fitstable_n_extensions(in);
    logverb("N extensions: %i\n", Next);
    for (i=1; i<Next; i++) {
        qfits_header* hdr = fitstable_get_header(in);
        int width, rows;

        width = qfits_header_getint(hdr, "NAXIS1", 0);
        rows = qfits_header_getint(hdr, "NAXIS2", 0);

        if (N < rows)
            rows = N;

        fits_header_mod_int(hdr, "NAXIS2", rows, "number of rows in table");
        if (qfits_header_dump(hdr, fid)) {
            ERROR("Failed to write HDU %i header", i);
            return -1;
        }

        if (rows && width) {
            int offset = in->table->col[0].off_beg;
            if (pipe_file_offset(fin, offset, (size_t)rows * (size_t)width, fid) ||
                fits_pad_file(fid)) {
                ERROR("Failed to write HDU %i data", i);
                return -1;
            }
        }

        if (i < Next-1)
            if (fitstable_open_next_extension(in)) {
                ERROR("Failed to open extension %i", i+1);
                return -1;
            }
    }
	
    if (fclose(fid)) {
        ERROR("Failed to close output file %s", outfn);
        return -1;
    }

    fclose(fin);
    fitstable_close(in);
    //fitstable_close(out);
    return 0;
}

