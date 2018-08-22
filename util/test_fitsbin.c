/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stddef.h>

#include "fitsbin.h"
#include "fitsioutils.h"

#include "cutest.h"

static char* get_tmpfile(int i) {
    static char fn[256];
    sprintf(fn, "/tmp/test-fitsbin-%i", i);
    return fn;
}

void test_fitsbin_1(CuTest* ct) {
    fitsbin_t* in, *out;
    int i;
    int N = 6;
    double outdata[6];
    double* indata;
    char* fn;
    fitsbin_chunk_t chunk;

    fn = get_tmpfile(0);
    out = fitsbin_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);

    CuAssertIntEquals(ct, 0, fitsbin_write_primary_header(out));

    for (i=0; i<N; i++) {
        outdata[i] = i*i;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test1";
    chunk.itemsize = sizeof(double);
    chunk.nrows = N;
    chunk.data = outdata;

    CuAssertIntEquals(ct, 0, fitsbin_write_chunk(out, &chunk));
    CuAssertIntEquals(ct, fitsbin_fix_primary_header(out), 0);
    CuAssertIntEquals(ct, fitsbin_close(out), 0);

    fitsbin_chunk_clean(&chunk);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i*i);
    }

    in = fitsbin_open(fn);
    CuAssertPtrNotNull(ct, in);

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test1";

    CuAssertIntEquals(ct, 0, fitsbin_read_chunk(in, &chunk));
    CuAssertIntEquals(ct, sizeof(double), chunk.itemsize);
    CuAssertIntEquals(ct, N, chunk.nrows);
    indata = chunk.data;
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, 0, memcmp(outdata, indata, sizeof(outdata)));
    CuAssertIntEquals(ct, 0, fitsbin_close(in));
}

void test_fitsbin_2(CuTest* ct) {
    fitsbin_t* in, *out;
    int i;
    int N = 6;
    double outdata[6];
    double* indata;
    char* fn;
    fitsbin_chunk_t chunk;
    fitsbin_chunk_t* ch;

    fn = get_tmpfile(0);
    printf("Writing to %s\n", fn);
    out = fitsbin_open_for_writing(fn);
    CuAssertPtrNotNull(ct, out);

    CuAssertIntEquals(ct, 0, fitsbin_write_primary_header(out));

    for (i=0; i<N; i++) {
        outdata[i] = i*i;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test2";
    chunk.itemsize = 1;
    //chunk.nrows = N * sizeof(double);
    chunk.data = outdata;

    CuAssertIntEquals(ct, 0, fitsbin_write_chunk_header(out, &chunk));
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitsbin_write_items(out, &chunk, &outdata[i], sizeof(double)));
    }
    CuAssertIntEquals(ct, 0, fitsbin_fix_chunk_header(out, &chunk));

    fitsbin_chunk_reset(&chunk);
    chunk.tablename = "test2B";
    chunk.itemsize = sizeof(double);
    //chunk.nrows = N;
    chunk.data = outdata;

    CuAssertIntEquals(ct, 0, fitsbin_write_chunk_header(out, &chunk));
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitsbin_write_items(out, &chunk, outdata + (N-1) - i, 1));
    }
    CuAssertIntEquals(ct, 0, fitsbin_fix_chunk_header(out, &chunk));

    CuAssertIntEquals(ct, fitsbin_close(out), 0);

    fitsbin_chunk_clean(&chunk);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i*i);
    }

    in = fitsbin_open(fn);
    CuAssertPtrNotNull(ct, in);

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test2";
    fitsbin_add_chunk(in, &chunk);
    chunk.tablename = "test2B";
    fitsbin_add_chunk(in, &chunk);

    CuAssertIntEquals(ct, 0, fitsbin_read(in));

    ch = fitsbin_get_chunk(in, 0);
    CuAssertIntEquals(ct, 1, ch->itemsize);
    CuAssertIntEquals(ct, N * sizeof(double), ch->nrows);
    indata = ch->data;
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, 0, memcmp(outdata, indata, sizeof(outdata)));
 
    ch = fitsbin_get_chunk(in, 1);
    CuAssertIntEquals(ct, sizeof(double), ch->itemsize);
    CuAssertIntEquals(ct, N, ch->nrows);
    indata = ch->data;
    CuAssertPtrNotNull(ct, indata);
    for (i=0; i<N; i++) {
        CuAssertDblEquals(ct, outdata[N-1 - i], indata[i], 1e-10);
    }

    CuAssertIntEquals(ct, 0, fitsbin_close(in));
}



void test_inmemory_fitsbin_1(CuTest* ct) {
    fitsbin_t* fb;
    int i;
    int N = 6;
    double outdata[6];
    double* indata;
    fitsbin_chunk_t chunk;

    fb = fitsbin_open_in_memory();
    CuAssertPtrNotNull(ct, fb);

    CuAssertIntEquals(ct, 0, fitsbin_write_primary_header(fb));

    for (i=0; i<N; i++) {
        outdata[i] = i*i;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test1";
    chunk.itemsize = sizeof(double);
    chunk.nrows = N;
    chunk.data = outdata;

    CuAssertIntEquals(ct, 0, fitsbin_write_chunk(fb, &chunk));
    CuAssertIntEquals(ct, 0, fitsbin_fix_primary_header(fb));

    fitsbin_chunk_clean(&chunk);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i*i);
    }

    CuAssertIntEquals(ct, 0, fitsbin_switch_to_reading(fb));

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test1";

    CuAssertIntEquals(ct, 0, fitsbin_read_chunk(fb, &chunk));
    CuAssertIntEquals(ct, sizeof(double), chunk.itemsize);
    CuAssertIntEquals(ct, N, chunk.nrows);
    indata = chunk.data;
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, 0, memcmp(outdata, indata, sizeof(outdata)));

    CuAssertIntEquals(ct, 0, fitsbin_close(fb));
}

void test_inmemory_fitsbin_2(CuTest* ct) {
    fitsbin_t* fb;
    int i;
    int N = 6;
    double outdata[6];
    double* indata;
    fitsbin_chunk_t chunk;
    fitsbin_chunk_t* ch;

    fb = fitsbin_open_in_memory();
    CuAssertPtrNotNull(ct, fb);

    CuAssertIntEquals(ct, 0, fitsbin_write_primary_header(fb));

    for (i=0; i<N; i++) {
        outdata[i] = i*i;
    }

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test2";
    chunk.itemsize = 1;
    //chunk.nrows = N * sizeof(double);
    chunk.data = outdata;

    CuAssertIntEquals(ct, 0, fitsbin_write_chunk_header(fb, &chunk));
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitsbin_write_items(fb, &chunk, &outdata[i], sizeof(double)));
    }
    CuAssertIntEquals(ct, 0, fitsbin_fix_chunk_header(fb, &chunk));

    fitsbin_chunk_reset(&chunk);
    chunk.tablename = "test2B";
    chunk.itemsize = sizeof(double);
    //chunk.nrows = N;
    chunk.data = outdata;

    CuAssertIntEquals(ct, 0, fitsbin_write_chunk_header(fb, &chunk));
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitsbin_write_items(fb, &chunk, outdata + (N-1) - i, 1));
    }
    CuAssertIntEquals(ct, 0, fitsbin_fix_chunk_header(fb, &chunk));

    fitsbin_chunk_clean(&chunk);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i*i);
    }

    CuAssertIntEquals(ct, 0, fitsbin_switch_to_reading(fb));

    fitsbin_chunk_init(&chunk);
    chunk.tablename = "test2";
    fitsbin_add_chunk(fb, &chunk);
    chunk.tablename = "test2B";
    fitsbin_add_chunk(fb, &chunk);

    CuAssertIntEquals(ct, 0, fitsbin_read(fb));

    ch = fitsbin_get_chunk(fb, 0);
    CuAssertIntEquals(ct, 1, ch->itemsize);
    CuAssertIntEquals(ct, N * sizeof(double), ch->nrows);
    indata = ch->data;
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, 0, memcmp(outdata, indata, sizeof(outdata)));
 
    ch = fitsbin_get_chunk(fb, 1);
    CuAssertIntEquals(ct, sizeof(double), ch->itemsize);
    CuAssertIntEquals(ct, N, ch->nrows);
    indata = ch->data;
    CuAssertPtrNotNull(ct, indata);
    for (i=0; i<N; i++) {
        CuAssertDblEquals(ct, outdata[N-1 - i], indata[i], 1e-10);
    }

    CuAssertIntEquals(ct, 0, fitsbin_close(fb));
}





