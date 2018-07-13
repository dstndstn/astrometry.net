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

#include "anqfits.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "errors.h"

int tabsort(const char* infn, const char* outfn, const char* colname,
            int descending) {
    FILE* fin;
    FILE* fout;
    int ext, nextens;
    off_t start, size;
    void* data = NULL;
    int* perm = NULL;
    unsigned char* map = NULL;
    size_t mapsize = 0;
    anqfits_t* anq = NULL;

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
    anq = anqfits_open(infn);
    if (!anq) {
        ERROR("Failed to open \"%s\"", infn);
        goto bailout;
    }
    start = anqfits_header_start(anq, 0);
    size  = anqfits_header_size (anq, 0);
    if (pipe_file_offset(fin, start, size, fout)) {
        ERROR("Failed to copy primary FITS header.");
        goto bailout;
    }

    nextens = anqfits_n_ext(anq);
    //logverb("Sorting %i extensions.\n", nextens);
    for (ext=1; ext<nextens; ext++) {
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
        off_t hdrstart, hdrsize, datsize, datstart;
        int i;

        hdrstart = anqfits_header_start(anq, ext);
        hdrsize  = anqfits_header_size (anq, ext);
        datstart = anqfits_data_start  (anq, ext);
        datsize  = anqfits_data_size   (anq, ext);
        if (!anqfits_is_table(anq, ext)) {
            ERROR("Extension %i isn't a table. Skipping.\n", ext);
            continue;
        }
        table = anqfits_get_table(anq, ext);
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
        case TFITS_BIN_TYPE_K:
            data = realloc(data, table->nr * sizeof(int64_t));
            if (descending)
                sort_func = compare_int64_desc;
            else
                sort_func = compare_int64_asc;
            break;

        default:
            ERROR("Column %s is neither FITS type D, E, nor K.  Skipping.", colname);
            continue;
        }

        // Grab the sort column.
        atomsize = fits_get_atom_size(col->atom_type);
        printf("Reading sort column \"%s\"\n", colname);
        qfits_query_column_seq_to_array(table, c, 0, table->nr, data, atomsize);
        // Sort it.
        printf("Sorting sort column\n");
        perm = permuted_sort(data, atomsize, sort_func, NULL, table->nr);

        // mmap the input file.
        printf("mmapping input file\n");
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
        tabledata = map + (off_t)mgap + (datstart - hdrstart);
        tablehdr  = map + (off_t)mgap;

        // Copy the table header without change.
        printf("Copying table header.\n");
        if (fwrite(tablehdr, 1, hdrsize, fout) != hdrsize) {
            SYSERROR("Failed to write FITS table header");
            goto bailout;
        }

        for (i=0; i<table->nr; i++) {
            unsigned char* rowptr;
            if (i % 100000 == 0)
                printf("Writing row %i\n", i);
            rowptr = tabledata + (off_t)(perm[i]) * (off_t)table->tab_w;
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
    anqfits_close(anq);
    printf("Done\n");
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

