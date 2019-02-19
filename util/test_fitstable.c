/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stddef.h>

#include "fitstable.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "an-endian.h"
#include "qfits_header.h"

#include "cutest.h"

static char* get_tmpfile(int i) {
    static char fn[256];
    sprintf(fn, "/tmp/test-fitstable-%i", i);
    return fn;
}

static void print_hex(const void* v, int N) {
    int i;
    for (i=0; i<N; i++) {
        printf("%02x ", ((unsigned char*)v)[i]);
        if (i%4 == 3)
            printf(", ");
    }
}

void test_copy_rows_file_to_memory(CuTest* ct) {
    fitstable_t *t1, *t2;
    tfits_type dubl;
    tfits_type flot;
    double* d1;
    double* d2;
    double* d3;
    char* name1 = "RA";
    char* name2 = "DEC";
    char* name3 = "SORT";
    int i, N, N1;
    int rtn;
    int* order;
    char* fn1 = get_tmpfile(42);

    double d1in[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    double d2in[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    double d3in[10] = { 4, 5, 6, 7, 0, 1, 8, 9, 2, 3 };

    printf("Big endian: %i\n", is_big_endian());
    printf("qfits thinks: %i\n", qfits_is_platform_big_endian());
    CuAssertIntEquals(ct, is_big_endian(), qfits_is_platform_big_endian());

    fits_use_error_system();

    N = sizeof(d1in) / sizeof(double);
    for (i=0; i<N; i++)
        d2in[i] += 1000.;

    dubl = fitscolumn_double_type();
    flot = fitscolumn_float_type();
    t1 = fitstable_open_for_writing(fn1);
    t2 = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, t1);
    CuAssertPtrNotNull(ct, t2);

    fitstable_add_write_column(t1, dubl, name1, "u1");
    fitstable_add_write_column(t1, dubl, name2, "u2");
    fitstable_add_write_column_convert(t1, flot, dubl, name3, "u3");

    rtn = fitstable_write_primary_header(t1);
    CuAssertIntEquals(ct, rtn, 0);
    rtn = fitstable_write_header(t1);
    CuAssertIntEquals(ct, rtn, 0);

    for (i=0; i<N; i++)
        rtn = fitstable_write_row(t1, d1in+i, d2in+i, d3in+i);

    rtn = fitstable_fix_header(t1);
    CuAssertIntEquals(ct, rtn, 0);

    fitstable_print_columns(t1);

    rtn = fitstable_close(t1);
    CuAssertIntEquals(ct, rtn, 0);
    printf("Wrote to file:  %s\n", fn1);

    // 'tablist' etc show that the file is good (has been endian-flipped).

    // now re-open that file.
    t1 = fitstable_open(fn1);
    CuAssertPtrNotNull(ct, t1);
	
    d1 = fitstable_read_column(t1, name1, dubl);
    d2 = fitstable_read_column(t1, name2, dubl);
    d3 = fitstable_read_column(t1, name3, dubl);
    CuAssertPtrNotNull(ct, d1);
    CuAssertPtrNotNull(ct, d2);
    CuAssertPtrNotNull(ct, d3);

    printf("\nT1:\n");
    for (i=0; i<N; i++) {
        printf("%g %g %g\n", d1[i], d2[i], d3[i]);
    }

    N1 = fitstable_nrows(t1);
    CuAssertIntEquals(ct, N1, N);
    order = permuted_sort(d3, sizeof(double), compare_doubles_asc, NULL, N1);
    CuAssertPtrNotNull(ct, order);

    fitstable_add_fits_columns_as_struct2(t1, t2);
    rtn = fitstable_write_header(t2);
    CuAssertIntEquals(ct, rtn, 0);

    rtn = fitstable_row_size(t1);
    // 2 * D + E
    CuAssertIntEquals(ct, rtn, 8 + 8 + 4);
    rtn = fitstable_row_size(t2);
    CuAssertIntEquals(ct, rtn, 8 + 8 + 4);

    fitstable_add_fits_columns_as_struct(t1);
    rtn = fitstable_copy_rows_data(t1, order, N1, t2);
    CuAssertIntEquals(ct, rtn, 0);
	
    rtn = fitstable_fix_header(t2);
    CuAssertIntEquals(ct, rtn, 0);

    free(order);
    fitstable_close(t1);

    rtn = fitstable_switch_to_reading(t2);
    CuAssertIntEquals(ct, rtn, 0);

    free(d1);
    free(d2);
    free(d3);

    d1 = fitstable_read_column(t2, name1, dubl);
    d2 = fitstable_read_column(t2, name2, dubl);
    d3 = fitstable_read_column(t2, name3, dubl);
    CuAssertPtrNotNull(ct, d1);
    CuAssertPtrNotNull(ct, d2);
    CuAssertPtrNotNull(ct, d3);

    printf("\nT2:\n");
    for (i=0; i<N; i++) {
        printf("%g %g %g\n", d1[i], d2[i], d3[i]);
    }

    // expected values
    double ex1[] = { 4,5,8,9,0,1,2,3,6,7 };
    double ex2[] = { 1004,1005,1008,1009,1000,1001,1002,1003,1006,1007 };
    double ex3[] = { 0,1,2,3,4,5,6,7,8,9 };
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, ex1[i], d1[i]);
        CuAssertIntEquals(ct, ex2[i], d2[i]);
        CuAssertIntEquals(ct, ex3[i], d3[i]);
    }

    fitstable_close(t2);

    free(d1);
    free(d2);
    free(d3);
}


void test_copy_rows_inmemory(CuTest* ct) {
    fitstable_t *t1, *t2;
    tfits_type dubl;
    tfits_type flot;
    double* d1;
    double* d2;
    double* d3;
    char* name1 = "RA";
    char* name2 = "DEC";
    char* name3 = "SORT";
    int i, N, N1;
    int rtn;
    int* order;

    double d1in[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    double d2in[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    double d3in[10] = { 4, 5, 6, 7, 0, 1, 8, 9, 2, 3 };

    N = sizeof(d1in) / sizeof(double);
    for (i=0; i<N; i++)
        d2in[i] += 1000.;

    dubl = fitscolumn_double_type();
    flot = fitscolumn_float_type();
    t1 = fitstable_open_in_memory();
    t2 = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, t1);
    CuAssertPtrNotNull(ct, t2);

    fitstable_add_write_column(t1, dubl, name1, "u1");
    fitstable_add_write_column(t1, dubl, name2, "u2");
    fitstable_add_write_column_convert(t1, flot, dubl, name3, "u3");
    rtn = fitstable_write_header(t1);
    CuAssertIntEquals(ct, rtn, 0);

    for (i=0; i<N; i++)
        rtn = fitstable_write_row(t1, d1in+i, d2in+i, d3in+i);

    rtn = fitstable_fix_header(t1);
    CuAssertIntEquals(ct, rtn, 0);

    qfits_header_list(fitstable_get_header(t1), stdout);

    fitstable_print_columns(t1);

    printf("d1in[1]  ");
    print_hex(d1in+1, sizeof(double));
    printf("\n");
    printf("d2in[1]  ");
    print_hex(d2in+1, sizeof(double));
    printf("\n");
    printf("d3in[1]  ");
    print_hex(d3in+1, sizeof(double));
    printf("\n");
    float f = d3in[1];
    printf("float(d3in[1])  ");
    print_hex(&f, sizeof(float));
    printf("\n");

    printf("row0:  ");
    print_hex(bl_access(t1->rows, 1), bl_datasize(t1->rows));
    printf("\n");

    rtn = fitstable_switch_to_reading(t1);
    CuAssertIntEquals(ct, rtn, 0);
	
    d1 = fitstable_read_column(t1, name1, dubl);
    d2 = fitstable_read_column(t1, name2, dubl);
    d3 = fitstable_read_column(t1, name3, dubl);
    CuAssertPtrNotNull(ct, d1);
    CuAssertPtrNotNull(ct, d2);
    CuAssertPtrNotNull(ct, d3);

    for (i=0; i<N; i++) {
        printf("%g %g %g\n", d1[i], d2[i], d3[i]);
    }

    N1 = fitstable_nrows(t1);
    CuAssertIntEquals(ct, N1, N);
    order = permuted_sort(d3, sizeof(double), compare_doubles_asc, NULL, N1);
    CuAssertPtrNotNull(ct, order);

    fitstable_add_fits_columns_as_struct2(t1, t2);
    rtn = fitstable_write_header(t2);
    CuAssertIntEquals(ct, rtn, 0);

    rtn = fitstable_row_size(t1);
    // 2 * D + E
    CuAssertIntEquals(ct, rtn, 8 + 8 + 4);
    rtn = fitstable_row_size(t2);
    CuAssertIntEquals(ct, rtn, 8 + 8 + 4);

    rtn = fitstable_copy_rows_data(t1, order, N1, t2);
    CuAssertIntEquals(ct, rtn, 0);
	
    rtn = fitstable_fix_header(t2);
    CuAssertIntEquals(ct, rtn, 0);

    free(order);
    fitstable_close(t1);

    rtn = fitstable_switch_to_reading(t2);
    CuAssertIntEquals(ct, rtn, 0);

    free(d1);
    free(d2);
    free(d3);

    d1 = fitstable_read_column(t2, name1, dubl);
    d2 = fitstable_read_column(t2, name2, dubl);
    d3 = fitstable_read_column(t2, name3, dubl);
    CuAssertPtrNotNull(ct, d1);
    CuAssertPtrNotNull(ct, d2);
    CuAssertPtrNotNull(ct, d3);

    printf("\nT2:\n");
    for (i=0; i<N; i++) {
        printf("%g %g %g\n", d1[i], d2[i], d3[i]);
    }

    // expected values
    double ex1[] = { 4,5,8,9,0,1,2,3,6,7 };
    double ex2[] = { 1004,1005,1008,1009,1000,1001,1002,1003,1006,1007 };
    double ex3[] = { 0,1,2,3,4,5,6,7,8,9 };
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, ex1[i], d1[i]);
        CuAssertIntEquals(ct, ex2[i], d2[i]);
        CuAssertIntEquals(ct, ex3[i], d3[i]);
    }

    fitstable_close(t2);

    free(d1);
    free(d2);
    free(d3);
}



void test_one_column_write_read(CuTest* ct) {
    fitstable_t* tab, *outtab;
    int i;
    int N = 6;
    double outdata[6];
    double* indata;
    char* fn;
    tfits_type dubl;

    fn = get_tmpfile(0);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);

    dubl = fitscolumn_double_type();
    
    fitstable_add_write_column(outtab, dubl, "X", "foounits");
    CuAssertIntEquals(ct, fitstable_ncols(outtab), 1);

    CuAssertIntEquals(ct, fitstable_write_primary_header(outtab), 0);
    CuAssertIntEquals(ct, fitstable_write_header(outtab), 0);

    for (i=0; i<N; i++) {
        outdata[i] = i*i;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fitstable_write_row(outtab, outdata+i), 0);
    }
    CuAssertIntEquals(ct, fitstable_fix_header(outtab), 0);
    CuAssertIntEquals(ct, fitstable_close(outtab), 0);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i*i);
    }

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    indata = fitstable_read_column(tab, "X", dubl);
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, 0, memcmp(outdata, indata, sizeof(outdata)));
    free(indata);
    CuAssertIntEquals(ct, 0, fitstable_close(tab));

}

void test_headers(CuTest* ct) {
    fitstable_t* tab, *outtab;
    char* fn;
    qfits_header* hdr;

    fn = get_tmpfile(1);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);

    hdr = fitstable_get_primary_header(outtab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR", 42, "Test Comment");

    // [add columns...]

    hdr = fitstable_get_header(outtab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR2", 99, "Test 2 Comment");

    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));
    
    CuAssertIntEquals(ct, 0, fitstable_ncols(outtab));

    CuAssertIntEquals(ct, 0, fitstable_close(outtab));

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);

    hdr = fitstable_get_primary_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 42, qfits_header_getint(hdr, "TSTHDR", -1));

    hdr = fitstable_get_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 99, qfits_header_getint(hdr, "TSTHDR2", -1));

    CuAssertIntEquals(ct, fitstable_close(tab), 0);
}

void test_multi_headers(CuTest* ct) {
    fitstable_t* tab, *outtab;
    char* fn;
    qfits_header* hdr;

    fn = get_tmpfile(2);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);

    hdr = fitstable_get_primary_header(outtab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR", 42, "Test Comment");

    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));

    // [add columns...]

    hdr = fitstable_get_header(outtab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR2", 99, "Test 2 Comment");
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_ncols(outtab));

    fitstable_next_extension(outtab);
    // [add columns...]
    hdr = fitstable_get_header(outtab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR3", 101, "Test 3 Comment");
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_ncols(outtab));

    fitstable_next_extension(outtab);
    // [add columns...]
    hdr = fitstable_get_header(outtab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR4", 104, "Test 4 Comment");
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_ncols(outtab));

    CuAssertIntEquals(ct, 0, fitstable_close(outtab));

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);

    hdr = fitstable_get_primary_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 42, qfits_header_getint(hdr, "TSTHDR", -1));

    hdr = fitstable_get_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 99, qfits_header_getint(hdr, "TSTHDR2", -1));

    fitstable_open_next_extension(tab);
    hdr = fitstable_get_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 101, qfits_header_getint(hdr, "TSTHDR3", -1));

    fitstable_open_next_extension(tab);
    hdr = fitstable_get_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 104, qfits_header_getint(hdr, "TSTHDR4", -1));

    CuAssertIntEquals(ct, fitstable_close(tab), 0);
}

void test_one_int_column_write_read(CuTest* ct) {
    fitstable_t* tab, *outtab;
    int i;
    int N = 100;
    int32_t outdata[N];
    int32_t* indata;
    char* fn;
    tfits_type i32 = TFITS_BIN_TYPE_J;

    fn = get_tmpfile(3);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);

    fitstable_add_write_column(outtab, i32, "X", "foounits");
    CuAssertIntEquals(ct, fitstable_ncols(outtab), 1);

    CuAssertIntEquals(ct, fitstable_write_primary_header(outtab), 0);
    CuAssertIntEquals(ct, fitstable_write_header(outtab), 0);

    for (i=0; i<N; i++) {
        outdata[i] = i;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fitstable_write_row(outtab, outdata+i), 0);
    }
    CuAssertIntEquals(ct, fitstable_fix_header(outtab), 0);
    CuAssertIntEquals(ct, fitstable_close(outtab), 0);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i);
    }

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    indata = fitstable_read_column(tab, "X", i32);
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, memcmp(outdata, indata, sizeof(outdata)), 0);
    free(indata);
    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_two_columns_with_conversion(CuTest* ct) {
    fitstable_t* tab, *outtab;
    int i;
    int N = 100;
    int32_t outx[N];
    double outy[N];
    double* inx;
    int32_t* iny;
    char* fn;

    tfits_type i32 = TFITS_BIN_TYPE_J;
    tfits_type dubl = fitscolumn_double_type();

    fn = get_tmpfile(4);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);

    fitstable_add_write_column(outtab, i32,  "X", "foounits");
    fitstable_add_write_column(outtab, dubl, "Y", "foounits");
    CuAssertIntEquals(ct, 2, fitstable_ncols(outtab));

    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));

    for (i=0; i<N; i++) {
        outx[i] = i;
        outy[i] = i+1000;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitstable_write_row(outtab, outx+i, outy+i));
    }
    CuAssertIntEquals(ct, 0, fitstable_fix_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_close(outtab));

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, i, outx[i]);
        CuAssertIntEquals(ct, i+1000, outy[i]);
    }

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    inx = fitstable_read_column(tab, "X", dubl);
    CuAssertPtrNotNull(ct, inx);
    iny = fitstable_read_column(tab, "Y", i32);
    CuAssertPtrNotNull(ct, iny);

    for (i=0; i<N; i++) {
        /*
         printf("inx %g, outx %i, iny %i, outy %g\n",
         inx[i], outx[i], iny[i], outy[i]);
         */
        CuAssertIntEquals(ct, outx[i], (int)inx[i]);
        CuAssertIntEquals(ct, (int)outy[i], iny[i]);
    }

    free(inx);
    free(iny);

    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_arrays(CuTest* ct) {
    fitstable_t* tab, *outtab;
    int i;
    int N = 100;
    int DX = 4;
    int DY = 3;
    int32_t outx[N*DX];
    double outy[N*DY];
    double* inx;
    int32_t* iny;
    char* fn;
    int d, t;

    tfits_type i32 = TFITS_BIN_TYPE_J;
    tfits_type dubl = fitscolumn_double_type();

    fn = get_tmpfile(5);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));

    // first extension: arrays

    fitstable_add_write_column_array(outtab, i32,  DX, "X", "foounits");
    fitstable_add_write_column_array(outtab, dubl, DY, "Y", "foounits");
    CuAssertIntEquals(ct, 2, fitstable_ncols(outtab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));

    for (i=0; i<N*DX; i++)
        outx[i] = i;
    for (i=0; i<N*DY; i++)
        outy[i] = i+1000;

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitstable_write_row(outtab, outx+i*DX, outy+i*DY));
    }
    CuAssertIntEquals(ct, 0, fitstable_fix_header(outtab));

    // writing shouldn't affect the data values
    for (i=0; i<N*DX; i++)
        CuAssertIntEquals(ct, i, outx[i]);
    for (i=0; i<N*DY; i++)
        CuAssertIntEquals(ct, i+1000, outy[i]);


    // second extension: scalars
    fitstable_next_extension(outtab);
    fitstable_clear_table(outtab);

    fitstable_add_write_column(outtab, i32,  "X", "foounits");
    fitstable_add_write_column(outtab, dubl, "Y", "foounits");
    CuAssertIntEquals(ct, 2, fitstable_ncols(outtab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));

    for (i=0; i<N; i++)
        CuAssertIntEquals(ct, 0, fitstable_write_row(outtab, outx+i, outy+i));
    CuAssertIntEquals(ct, 0, fitstable_fix_header(outtab));

    // writing shouldn't affect the data values
    for (i=0; i<N*DX; i++)
        CuAssertIntEquals(ct, i, outx[i]);
    for (i=0; i<N*DY; i++)
        CuAssertIntEquals(ct, i+1000, outy[i]);

    CuAssertIntEquals(ct, 0, fitstable_close(outtab));



    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X");
    t = fitstable_get_type(tab, "X");

    CuAssertIntEquals(ct, DX, d);
    CuAssertIntEquals(ct, i32, t);

    d = fitstable_get_array_size(tab, "Y");
    t = fitstable_get_type(tab, "Y");

    CuAssertIntEquals(ct, DY, d);
    CuAssertIntEquals(ct, dubl, t);

    inx = fitstable_read_column_array(tab, "X", dubl);
    CuAssertPtrNotNull(ct, inx);
    iny = fitstable_read_column_array(tab, "Y", i32);
    CuAssertPtrNotNull(ct, iny);

    for (i=0; i<N*DX; i++)
        CuAssertIntEquals(ct, outx[i], (int)inx[i]);
    for (i=0; i<N*DY; i++)
        CuAssertIntEquals(ct, (int)outy[i], iny[i]);

    free(inx);
    free(iny);

    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_conversion(CuTest* ct) {
    fitstable_t* tab, *outtab;
    int i;
    int N = 100;
    int D = 3;
    double out[N*D];
    double* in;
    char* fn;
    int d, t;

    tfits_type i32 = TFITS_BIN_TYPE_J;
    tfits_type dubl = fitscolumn_double_type();

    fn = get_tmpfile(6);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));

    fitstable_add_write_column_array_convert(outtab, i32,  dubl, D, "X", "foounits");
    CuAssertIntEquals(ct, 1, fitstable_ncols(outtab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));

    for (i=0; i<N*D; i++)
        out[i] = i;

    for (i=0; i<N; i++)
        CuAssertIntEquals(ct, 0, fitstable_write_row(outtab, out+i*D));
    CuAssertIntEquals(ct, 0, fitstable_fix_header(outtab));

    // writing shouldn't affect the data values
    for (i=0; i<N*D; i++)
        CuAssertIntEquals(ct, i, out[i]);

    CuAssertIntEquals(ct, 0, fitstable_close(outtab));

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X");
    t = fitstable_get_type(tab, "X");

    CuAssertIntEquals(ct, D, d);
    CuAssertIntEquals(ct, i32, t);

    in = fitstable_read_column_array(tab, "X", dubl);
    CuAssertPtrNotNull(ct, in);

    for (i=0; i<N*D; i++)
        CuAssertIntEquals(ct, out[i], (int)in[i]);
    free(in);
    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

struct ts1 {
    int x1;
    int x2[3];
    double x3;
    double x4;
};
typedef struct ts1 ts1;

struct ts2 {
    double x1;
    int16_t x2[3];
    int x3;
    float x4;
};
typedef struct ts2 ts2;

void test_struct_1(CuTest* ct) {
    fitstable_t* tab, *outtab;
    tfits_type i16 = TFITS_BIN_TYPE_I;
    tfits_type itype = fitscolumn_int_type();
    tfits_type dubl = fitscolumn_double_type();
    tfits_type flt = fitscolumn_float_type();
    tfits_type anytype = fitscolumn_any_type();
    char* fn;
    int i, N = 100;
    ts1 x[N];
    ts2 y[N];
    int d, t;

    fn = get_tmpfile(7);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));

    fitstable_add_write_column_struct(outtab, itype, 1, offsetof(ts1, x1),
                                      itype, "X1", "x1units");
    fitstable_add_write_column_struct(outtab, itype, 3, offsetof(ts1, x2),
                                      i16, "X2", "x2units");
    fitstable_add_write_column_struct(outtab, dubl, 1, offsetof(ts1, x3),
                                      dubl, "X3", "x3units");
    fitstable_add_write_column_struct(outtab, dubl, 1, offsetof(ts1, x4),
                                      flt, "X4", "x4units");
    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));

    for (i=0; i<N; i++) {
        x[i].x1 = i;
        x[i].x2[0] = 1000 + i;
        x[i].x2[1] = 2000 + i;
        x[i].x2[2] = 3000 + i;
        x[i].x3 = i * 1000.0;
        x[i].x4 = i * 1e6;
        CuAssertIntEquals(ct, 0, fitstable_write_struct(outtab, x+i));
    }

    CuAssertIntEquals(ct, 0, fitstable_fix_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_close(outtab));

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X1");
    t = fitstable_get_type(tab, "X1");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, itype, t);

    d = fitstable_get_array_size(tab, "X2");
    t = fitstable_get_type(tab, "X2");
    CuAssertIntEquals(ct, 3, d);
    CuAssertIntEquals(ct, i16, t);

    d = fitstable_get_array_size(tab, "X3");
    t = fitstable_get_type(tab, "X3");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, dubl, t);

    d = fitstable_get_array_size(tab, "X4");
    t = fitstable_get_type(tab, "X4");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, flt, t);

    fitstable_add_read_column_struct(tab, dubl, 1, offsetof(ts2, x1),
                                     anytype, "X1", TRUE);
    fitstable_add_read_column_struct(tab, i16, 3, offsetof(ts2, x2),
                                     anytype, "X2", TRUE);
    fitstable_add_read_column_struct(tab, itype, 1, offsetof(ts2, x3),
                                     anytype, "X3", TRUE);
    fitstable_add_read_column_struct(tab, flt, 1, offsetof(ts2, x4),
                                     anytype, "X4", TRUE);

    fitstable_read_extension(tab, 1);

    for (i=0; i<N; i++) {
        double eps = 1e-10;
        CuAssertIntEquals(ct, 0, fitstable_read_struct(tab, i, y+i));
        CuAssertDblEquals(ct, x[i].x1, y[i].x1, eps);
        CuAssertIntEquals(ct, x[i].x2[0], y[i].x2[0]);
        CuAssertIntEquals(ct, x[i].x2[1], y[i].x2[1]);
        CuAssertIntEquals(ct, x[i].x2[2], y[i].x2[2]);
        CuAssertIntEquals(ct, x[i].x3, y[i].x3);
        CuAssertDblEquals(ct, x[i].x4, y[i].x4, eps);
    }

    memset(y, 0, sizeof(y));

    CuAssertIntEquals(ct, 0, fitstable_read_structs(tab, y, sizeof(ts2), 0, N));
    for (i=0; i<N; i++) {
        double eps = 1e-10;
        CuAssertDblEquals(ct, x[i].x1, y[i].x1, eps);
        CuAssertIntEquals(ct, x[i].x2[0], y[i].x2[0]);
        CuAssertIntEquals(ct, x[i].x2[1], y[i].x2[1]);
        CuAssertIntEquals(ct, x[i].x2[2], y[i].x2[2]);
        CuAssertIntEquals(ct, x[i].x3, y[i].x3);
        CuAssertDblEquals(ct, x[i].x4, y[i].x4, eps);
    }
    CuAssertIntEquals(ct, 0, fitstable_close(tab));

}


struct ts3 {
    double x1;
    int16_t x2[3];
    int x3;
    float x4;
};
typedef struct ts3 ts3;

static void add_columns(fitstable_t* tab, anbool writing) {
    tfits_type i16 = TFITS_BIN_TYPE_I;
    tfits_type itype = fitscolumn_int_type();
    tfits_type dubl = fitscolumn_double_type();
    tfits_type flt = fitscolumn_float_type();
    tfits_type anytype = fitscolumn_any_type();

    fitstable_add_column_struct(tab, dubl, 1, offsetof(ts3, x1),
                                (writing ? itype : anytype), "X1",
                                "x1units", TRUE);
    fitstable_add_column_struct(tab, i16, 3, offsetof(ts3, x2),
                                (writing ? itype : anytype), "X2",
                                "x2units", TRUE);
    fitstable_add_column_struct(tab, itype, 1, offsetof(ts3, x3),
                                (writing ? flt : anytype), "X3",
                                "x3units", TRUE);
    fitstable_add_column_struct(tab, flt, 1, offsetof(ts3, x4),
                                (writing ? dubl : anytype), "X4",
                                "x4units", TRUE);
}

void test_struct_2(CuTest* ct) {
    fitstable_t* tab, *outtab;
    char* fn;
    int i, N = 100;
    ts3 x[N];
    ts3 y[N];
    int d, t;
    tfits_type itype = fitscolumn_int_type();
    tfits_type dubl = fitscolumn_double_type();
    tfits_type flt = fitscolumn_float_type();

    fn = get_tmpfile(8);
    outtab = fitstable_open_for_writing(fn);
    CuAssertPtrNotNull(ct, outtab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(outtab));

    add_columns(outtab, TRUE);

    CuAssertIntEquals(ct, 0, fitstable_write_header(outtab));

    for (i=0; i<N; i++) {
        x[i].x1 = i;
        x[i].x2[0] = 10 + i;
        x[i].x2[1] = 20 + i;
        x[i].x2[2] = 30 + i;
        x[i].x3 = i * 1000;
        x[i].x4 = i * 1e6;
        CuAssertIntEquals(ct, 0, fitstable_write_struct(outtab, x+i));
    }

    CuAssertIntEquals(ct, 0, fitstable_fix_header(outtab));
    CuAssertIntEquals(ct, 0, fitstable_close(outtab));

    tab = fitstable_open(fn);
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X1");
    t = fitstable_get_type(tab, "X1");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, itype, t);

    d = fitstable_get_array_size(tab, "X2");
    t = fitstable_get_type(tab, "X2");
    CuAssertIntEquals(ct, 3, d);
    CuAssertIntEquals(ct, itype, t);

    d = fitstable_get_array_size(tab, "X3");
    t = fitstable_get_type(tab, "X3");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, flt, t);

    d = fitstable_get_array_size(tab, "X4");
    t = fitstable_get_type(tab, "X4");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, dubl, t);

    add_columns(tab, FALSE);

    fitstable_read_extension(tab, 1);

    CuAssertIntEquals(ct, 0, fitstable_read_structs(tab, y, sizeof(ts3), 0, N));
    for (i=0; i<N; i++) {
        double eps = 1e-10;
        CuAssertDblEquals(ct, x[i].x1, y[i].x1, eps);
        CuAssertIntEquals(ct, x[i].x2[0], y[i].x2[0]);
        CuAssertIntEquals(ct, x[i].x2[1], y[i].x2[1]);
        CuAssertIntEquals(ct, x[i].x2[2], y[i].x2[2]);
        CuAssertIntEquals(ct, x[i].x3, y[i].x3);
        CuAssertDblEquals(ct, x[i].x4, y[i].x4, eps);
    }
    CuAssertIntEquals(ct, 0, fitstable_close(tab));

}



///////////////////////    in-memory versions   /////////////////////////



void test_inmemory_one_column_write_read(CuTest* ct) {
    fitstable_t* tab;
    int i;
    int N = 6;
    double outdata[6];
    double* indata;
    tfits_type dubl;

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);

    dubl = fitscolumn_double_type();
    
    fitstable_add_write_column(tab, dubl, "X", "foounits");
    CuAssertIntEquals(ct, fitstable_ncols(tab), 1);

    CuAssertIntEquals(ct, fitstable_write_primary_header(tab), 0);
    CuAssertIntEquals(ct, fitstable_write_header(tab), 0);

    for (i=0; i<N; i++) {
        outdata[i] = i*i;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fitstable_write_row(tab, outdata+i), 0);
    }
    CuAssertIntEquals(ct, fitstable_fix_header(tab), 0);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i*i);
    }

    // switch to reading...
    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));
    indata = fitstable_read_column(tab, "X", dubl);
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, 0, memcmp(outdata, indata, sizeof(outdata)));
    free(indata);
    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_inmemory_headers(CuTest* ct) {
    fitstable_t* tab;
    qfits_header* hdr;

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);

    hdr = fitstable_get_primary_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR", 42, "Test Comment");

    // [add columns...]

    hdr = fitstable_get_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    fits_header_add_int(hdr, "TSTHDR2", 99, "Test 2 Comment");

    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(tab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));
    
    CuAssertIntEquals(ct, 0, fitstable_ncols(tab));

    // READING
    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));

    hdr = fitstable_get_primary_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 42, qfits_header_getint(hdr, "TSTHDR", -1));

    hdr = fitstable_get_header(tab);
    CuAssertPtrNotNull(ct, hdr);
    CuAssertIntEquals(ct, 99, qfits_header_getint(hdr, "TSTHDR2", -1));

    CuAssertIntEquals(ct, fitstable_close(tab), 0);
}

void test_inmemory_one_int_column_write_read(CuTest* ct) {
    fitstable_t* tab;
    int i;
    int N = 100;
    int32_t outdata[N];
    int32_t* indata;
    tfits_type i32 = TFITS_BIN_TYPE_J;

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);

    fitstable_add_write_column(tab, i32, "X", "foounits");
    CuAssertIntEquals(ct, fitstable_ncols(tab), 1);

    CuAssertIntEquals(ct, fitstable_write_primary_header(tab), 0);
    CuAssertIntEquals(ct, fitstable_write_header(tab), 0);

    for (i=0; i<N; i++) {
        outdata[i] = i;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, fitstable_write_row(tab, outdata+i), 0);
    }
    CuAssertIntEquals(ct, fitstable_fix_header(tab), 0);

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outdata[i], i);
    }

    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));
    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    indata = fitstable_read_column(tab, "X", i32);
    CuAssertPtrNotNull(ct, indata);
    CuAssertIntEquals(ct, memcmp(outdata, indata, sizeof(outdata)), 0);
    free(indata);
    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_inmemory_two_columns_with_conversion(CuTest* ct) {
    fitstable_t* tab;
    int i;
    int N = 100;
    int32_t outx[N];
    double outy[N];
    double* inx;
    int32_t* iny;

    tfits_type i32 = TFITS_BIN_TYPE_J;
    tfits_type dubl = fitscolumn_double_type();

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);

    fitstable_add_write_column(tab, i32,  "X", "foounits");
    fitstable_add_write_column(tab, dubl, "Y", "foounits");
    CuAssertIntEquals(ct, 2, fitstable_ncols(tab));

    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(tab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));

    for (i=0; i<N; i++) {
        outx[i] = i;
        outy[i] = i+1000;
    }

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitstable_write_row(tab, outx+i, outy+i));
    }
    CuAssertIntEquals(ct, 0, fitstable_fix_header(tab));

    // writing shouldn't affect the data values
    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, i, outx[i]);
        CuAssertIntEquals(ct, i+1000, outy[i]);
    }

    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    inx = fitstable_read_column(tab, "X", dubl);
    CuAssertPtrNotNull(ct, inx);
    iny = fitstable_read_column(tab, "Y", i32);
    CuAssertPtrNotNull(ct, iny);

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, outx[i], (int)inx[i]);
        CuAssertIntEquals(ct, (int)outy[i], iny[i]);
    }

    free(inx);
    free(iny);

    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_inmemory_struct_2(CuTest* ct) {
    fitstable_t* tab;
    int i, N = 100;
    ts3 x[N];
    ts3 y[N];
    int d, t;
    tfits_type itype = fitscolumn_int_type();
    tfits_type dubl = fitscolumn_double_type();
    tfits_type flt = fitscolumn_float_type();

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(tab));

    add_columns(tab, TRUE);

    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));

    for (i=0; i<N; i++) {
        x[i].x1 = i;
        x[i].x2[0] = 10 + i;
        x[i].x2[1] = 20 + i;
        x[i].x2[2] = 30 + i;
        x[i].x3 = i * 1000;
        x[i].x4 = i * 1e6;
        CuAssertIntEquals(ct, 0, fitstable_write_struct(tab, x+i));
    }

    CuAssertIntEquals(ct, 0, fitstable_fix_header(tab));

    // reading
    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X1");
    t = fitstable_get_type(tab, "X1");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, itype, t);

    d = fitstable_get_array_size(tab, "X2");
    t = fitstable_get_type(tab, "X2");
    CuAssertIntEquals(ct, 3, d);
    CuAssertIntEquals(ct, itype, t);

    d = fitstable_get_array_size(tab, "X3");
    t = fitstable_get_type(tab, "X3");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, flt, t);

    d = fitstable_get_array_size(tab, "X4");
    t = fitstable_get_type(tab, "X4");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, dubl, t);

    add_columns(tab, FALSE);

    fitstable_read_extension(tab, 1);

    CuAssertIntEquals(ct, 0, fitstable_read_structs(tab, y, sizeof(ts3), 0, N));
    for (i=0; i<N; i++) {
        double eps = 1e-10;
        CuAssertDblEquals(ct, x[i].x1, y[i].x1, eps);
        CuAssertIntEquals(ct, x[i].x2[0], y[i].x2[0]);
        CuAssertIntEquals(ct, x[i].x2[1], y[i].x2[1]);
        CuAssertIntEquals(ct, x[i].x2[2], y[i].x2[2]);
        CuAssertIntEquals(ct, x[i].x3, y[i].x3);
        CuAssertDblEquals(ct, x[i].x4, y[i].x4, eps);
    }
    CuAssertIntEquals(ct, 0, fitstable_close(tab));

}


void test_inmemory_arrays(CuTest* ct) {
    fitstable_t* tab;
    int i;
    int N = 100;
    int DX = 4;
    int DY = 3;
    int32_t outx[N*DX];
    double outy[N*DY];
    double* inx;
    int32_t* iny;
    int d, t;

    tfits_type i32 = TFITS_BIN_TYPE_J;
    tfits_type dubl = fitscolumn_double_type();

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(tab));

    // first extension: arrays

    fitstable_add_write_column_array(tab, i32,  DX, "X", "foounits");
    fitstable_add_write_column_array(tab, dubl, DY, "Y", "foounits");
    CuAssertIntEquals(ct, 2, fitstable_ncols(tab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));

    for (i=0; i<N*DX; i++)
        outx[i] = i;
    for (i=0; i<N*DY; i++)
        outy[i] = i+1000;

    for (i=0; i<N; i++) {
        CuAssertIntEquals(ct, 0, fitstable_write_row(tab, outx+i*DX, outy+i*DY));
    }
    CuAssertIntEquals(ct, 0, fitstable_fix_header(tab));

    // writing shouldn't affect the data values
    for (i=0; i<N*DX; i++)
        CuAssertIntEquals(ct, i, outx[i]);
    for (i=0; i<N*DY; i++)
        CuAssertIntEquals(ct, i+1000, outy[i]);

    // second extension: scalars
    fitstable_next_extension(tab);
    fitstable_clear_table(tab);

    fitstable_add_write_column(tab, i32,  "X", "foounits");
    fitstable_add_write_column(tab, dubl, "Y", "foounits");
    CuAssertIntEquals(ct, 2, fitstable_ncols(tab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));

    for (i=0; i<N; i++)
        CuAssertIntEquals(ct, 0, fitstable_write_row(tab, outx+i, outy+i));
    CuAssertIntEquals(ct, 0, fitstable_fix_header(tab));

    // writing shouldn't affect the data values
    for (i=0; i<N*DX; i++)
        CuAssertIntEquals(ct, i, outx[i]);
    for (i=0; i<N*DY; i++)
        CuAssertIntEquals(ct, i+1000, outy[i]);


    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X");
    t = fitstable_get_type(tab, "X");

    CuAssertIntEquals(ct, DX, d);
    CuAssertIntEquals(ct, i32, t);

    d = fitstable_get_array_size(tab, "Y");
    t = fitstable_get_type(tab, "Y");

    CuAssertIntEquals(ct, DY, d);
    CuAssertIntEquals(ct, dubl, t);

    inx = fitstable_read_column_array(tab, "X", dubl);
    CuAssertPtrNotNull(ct, inx);
    iny = fitstable_read_column_array(tab, "Y", i32);
    CuAssertPtrNotNull(ct, iny);

    for (i=0; i<N*DX; i++)
        CuAssertIntEquals(ct, outx[i], (int)inx[i]);
    for (i=0; i<N*DY; i++)
        CuAssertIntEquals(ct, (int)outy[i], iny[i]);
    free(inx);
    free(iny);
    inx = NULL;
    iny = NULL;

    fitstable_open_next_extension(tab);

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X");
    t = fitstable_get_type(tab, "X");

    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, i32, t);

    d = fitstable_get_array_size(tab, "Y");
    t = fitstable_get_type(tab, "Y");

    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, dubl, t);

    inx = fitstable_read_column_array(tab, "X", dubl);
    CuAssertPtrNotNull(ct, inx);
    iny = fitstable_read_column_array(tab, "Y", i32);
    CuAssertPtrNotNull(ct, iny);

    for (i=0; i<N; i++)
        CuAssertIntEquals(ct, outx[i], (int)inx[i]);
    for (i=0; i<N; i++)
        CuAssertIntEquals(ct, (int)outy[i], iny[i]);

    free(inx);
    free(iny);

    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_inmemory_conversion(CuTest* ct) {
    fitstable_t* tab;
    int i;
    int N = 100;
    int D = 3;
    double out[N*D];
    double* in;
    int d, t;

    tfits_type i32 = TFITS_BIN_TYPE_J;
    tfits_type dubl = fitscolumn_double_type();

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(tab));

    fitstable_add_write_column_array_convert(tab, i32,  dubl, D, "X", "foounits");
    CuAssertIntEquals(ct, 1, fitstable_ncols(tab));
    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));

    for (i=0; i<N*D; i++)
        out[i] = i;

    for (i=0; i<N; i++)
        CuAssertIntEquals(ct, 0, fitstable_write_row(tab, out+i*D));
    CuAssertIntEquals(ct, 0, fitstable_fix_header(tab));

    // writing shouldn't affect the data values
    for (i=0; i<N*D; i++)
        CuAssertIntEquals(ct, i, out[i]);

    // reading
    fitstable_clear_table(tab);

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X");
    t = fitstable_get_type(tab, "X");

    CuAssertIntEquals(ct, D, d);
    CuAssertIntEquals(ct, i32, t);

    in = fitstable_read_column_array(tab, "X", dubl);
    CuAssertPtrNotNull(ct, in);

    for (i=0; i<N*D; i++)
        CuAssertIntEquals(ct, out[i], (int)in[i]);
    free(in);
    CuAssertIntEquals(ct, 0, fitstable_close(tab));
}

void test_inmemory_struct_1(CuTest* ct) {
    fitstable_t* tab;
    tfits_type i16 = TFITS_BIN_TYPE_I;
    tfits_type itype = fitscolumn_int_type();
    tfits_type dubl = fitscolumn_double_type();
    tfits_type flt = fitscolumn_float_type();
    tfits_type anytype = fitscolumn_any_type();
    int i, N = 100;
    ts1 x[N];
    ts2 y[N];
    int d, t;

    tab = fitstable_open_in_memory();
    CuAssertPtrNotNull(ct, tab);
    CuAssertIntEquals(ct, 0, fitstable_write_primary_header(tab));

    fitstable_add_write_column_struct(tab, itype, 1, offsetof(ts1, x1),
                                      itype, "X1", "x1units");
    fitstable_add_write_column_struct(tab, itype, 3, offsetof(ts1, x2),
                                      i16, "X2", "x2units");
    fitstable_add_write_column_struct(tab, dubl, 1, offsetof(ts1, x3),
                                      dubl, "X3", "x3units");
    fitstable_add_write_column_struct(tab, dubl, 1, offsetof(ts1, x4),
                                      flt, "X4", "x4units");
    CuAssertIntEquals(ct, 0, fitstable_write_header(tab));

    for (i=0; i<N; i++) {
        x[i].x1 = i;
        x[i].x2[0] = 1000 + i;
        x[i].x2[1] = 2000 + i;
        x[i].x2[2] = 3000 + i;
        x[i].x3 = i * 1000.0;
        x[i].x4 = i * 1e6;
        CuAssertIntEquals(ct, 0, fitstable_write_struct(tab, x+i));
    }

    CuAssertIntEquals(ct, 0, fitstable_fix_header(tab));

	

    // reading
    CuAssertIntEquals(ct, 0, fitstable_switch_to_reading(tab));

    CuAssertIntEquals(ct, N, fitstable_nrows(tab));

    d = fitstable_get_array_size(tab, "X1");
    t = fitstable_get_type(tab, "X1");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, itype, t);

    d = fitstable_get_array_size(tab, "X2");
    t = fitstable_get_type(tab, "X2");
    CuAssertIntEquals(ct, 3, d);
    CuAssertIntEquals(ct, i16, t);

    d = fitstable_get_array_size(tab, "X3");
    t = fitstable_get_type(tab, "X3");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, dubl, t);

    d = fitstable_get_array_size(tab, "X4");
    t = fitstable_get_type(tab, "X4");
    CuAssertIntEquals(ct, 1, d);
    CuAssertIntEquals(ct, flt, t);

    fitstable_add_read_column_struct(tab, dubl, 1, offsetof(ts2, x1),
                                     anytype, "X1", TRUE);
    fitstable_add_read_column_struct(tab, i16, 3, offsetof(ts2, x2),
                                     anytype, "X2", TRUE);
    fitstable_add_read_column_struct(tab, itype, 1, offsetof(ts2, x3),
                                     anytype, "X3", TRUE);
    fitstable_add_read_column_struct(tab, flt, 1, offsetof(ts2, x4),
                                     anytype, "X4", TRUE);

    fitstable_read_extension(tab, 1);

    for (i=0; i<N; i++) {
        double eps = 1e-10;
        CuAssertIntEquals(ct, 0, fitstable_read_struct(tab, i, y+i));
        CuAssertDblEquals(ct, x[i].x1, y[i].x1, eps);
        CuAssertIntEquals(ct, x[i].x2[0], y[i].x2[0]);
        CuAssertIntEquals(ct, x[i].x2[1], y[i].x2[1]);
        CuAssertIntEquals(ct, x[i].x2[2], y[i].x2[2]);
        CuAssertIntEquals(ct, x[i].x3, y[i].x3);
        CuAssertDblEquals(ct, x[i].x4, y[i].x4, eps);
    }

    memset(y, 0, sizeof(y));

    CuAssertIntEquals(ct, 0, fitstable_read_structs(tab, y, sizeof(ts2), 0, N));
    for (i=0; i<N; i++) {
        double eps = 1e-10;
        CuAssertDblEquals(ct, x[i].x1, y[i].x1, eps);
        CuAssertIntEquals(ct, x[i].x2[0], y[i].x2[0]);
        CuAssertIntEquals(ct, x[i].x2[1], y[i].x2[1]);
        CuAssertIntEquals(ct, x[i].x2[2], y[i].x2[2]);
        CuAssertIntEquals(ct, x[i].x3, y[i].x3);
        CuAssertDblEquals(ct, x[i].x4, y[i].x4, eps);
    }
    CuAssertIntEquals(ct, 0, fitstable_close(tab));

}
