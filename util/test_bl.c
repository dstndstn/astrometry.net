/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "cutest.h"
#include "an-bool.h"
#include "bl.h"

void test_big_list(CuTest* tc) {
    // Test size_t sizes and indices of bl's.
    // Test ptrdiff_t as -1 or index
    CuAssertTrue(tc, -1 == BL_NOT_FOUND);
    CuAssertTrue(tc, BL_NOT_FOUND < 0);
    CuAssertTrue(tc, sizeof(size_t) == sizeof(ptrdiff_t));

    // Create a fake list so we don't have to allocate 16 GB of list to test.
    int blocksize = 1048576;
    il* big = il_new(blocksize);
    // that's 1<<20 ~ 1e6 elements per block

    // now need 1<<12 ~ 4096 blocks to overflow a 32-bit int
    int i;
    int N = 4096;
    bl_node* nodes = calloc(N, sizeof(bl_node));
    for (i=1; i<N; i++) {
        nodes[i-1].next = nodes+i;
        nodes[i-1].N = blocksize;
    }
    nodes[N-1].N = blocksize;
    big->head = nodes;
    big->tail = nodes + (N-1);
    big->N = (size_t)blocksize * (size_t)N;

    //bl_print_structure(big);

    printf("N %zu\n", il_size(big));
    printf("check: %s\n", (bl_check_consistency(big) ? "bad" : "ok"));

    int* p = il_append(big, 42);
    printf("appended: %i\n", *p);
    CuAssertIntEquals(tc, 42, *p);
    CuAssertTrue(tc, 4294967297L == il_size(big));

    size_t index = 4294967296L;
    int v = il_get(big, index);
    CuAssertIntEquals(tc, 42, v);

    big->last_access = NULL;
    big->last_access_n = 0;

    v = il_get(big, index);
    CuAssertIntEquals(tc, 42, v);

    for (i=0; i<blocksize*2; i++) {
        il_push(big, i);
    }

    il* split = il_new(1024);

    bl_split(big, split, (size_t)blocksize * (size_t)N);

    printf("split: %zu, %zu\n", il_size(big), il_size(split));
    CuAssertIntEquals(tc, 42, il_get(split, 0));
    CuAssertIntEquals(tc, 3, il_get(split, 4));

    il_append_list(big, split);

    CuAssertTrue(tc, 4297064449L == il_size(big));

    int* arr = calloc(1024, sizeof(int));

    bl_copy(big, (size_t)blocksize * (size_t)N, 1024, arr);

    CuAssertIntEquals(tc, 42, arr[0]);
    CuAssertIntEquals(tc,  3, arr[4]);
}



void test_il_sorted(CuTest* tc) {
    int vals[] = {34951,34950,34949,35049,35149,29951,29950,29949,34999,34998,5099,5199,39849,39949,4999,35249,29952,34952,5299,35349,29953,34953,5399,35449,29954,34954,5499,35549,29955,34955,5599,35649,29956,34956,5699,35749,29957,34957,5799,35849,29958,34958,5899,35949,29959,34959,5999,36049,29960,34960,6099,36149,29961,34961,6199,36249,29962,34962,6299,36349,29963,34963,6399,36449,29964,34964,6499,36549,29965,34965,6599,36649,29966,34966,6699,36749,29967,34967,6799,36849,29968,34968,6899,36949,29969,34969,6999,37049,29970,34970,7099,37149,29971,34971,7199,37249,29972,34972,7299,37349,29973,34973,7399,37449,29974,34974,7499,37549,29975,34975,7599,37649,29976,34976,7699,37749,29977,34977,7799,37849,29978,34978,7899,37949,29979,34979,7999,38049,29980,34980,8099,38149,29981,34981,8199,38249,29982,34982,8299,38349,29983,34983,8399,38449,29984,34984,8499,38549,29985,34985,8599,38649,29986,34986,8699,38749,29987,34987,8799,38849,29988,34988,8899,38949,29989,34989,8999,39049,29990,34990,9099,39149,29991,34991,9199,39249,29992,34992,9299,39349,29993,34993,9399,39449,29994,34994,9499,39549,29995,34995,9599,39649,29996,34996,9699,39749,29997,34997,9799,29998,9899,29999,9999};
    int i, N;
    il* lst;
    N = sizeof(vals)/sizeof(int);
    lst = il_new(256);
    for (i=0; i<N; i++)
        il_append(lst, vals[i]);
    CuAssertIntEquals(tc, N, il_size(lst));
    for (i=0; i<N; i++)
        CuAssertIntEquals(tc, vals[i], il_get(lst, i));

    printf("Before sorting:\n");
    il_print(lst);

    il_sort(lst, TRUE);

    printf("After sorting:\n");
    il_print(lst);

    CuAssertIntEquals(tc, 0, il_check_consistency(lst));
    CuAssertIntEquals(tc, 0, il_check_sorted_ascending(lst, TRUE));
    for (i=1; i<N; i++)
        CuAssertTrue(tc, il_get(lst, i-1) < il_get(lst, i));
    for (i=0; i<N; i++)
        CuAssertIntEquals(tc, 1, il_sorted_contains(lst, vals[i]));
}

void test_ll_sorted_contains_1(CuTest* tc) {
    int i;
    ll* lst = ll_new(32);
    // Fill the list with L[i] = i^2
    for (i=0; i<10000; i++)
        ll_append(lst, i*i);
    //ll_print(lst);
    CuAssertIntEquals(tc, 0, ll_sorted_contains(lst, -1));
    CuAssertIntEquals(tc, 1, ll_sorted_contains(lst, 0));
    CuAssertIntEquals(tc, 1, ll_sorted_contains(lst, 1));
    CuAssertIntEquals(tc, 1, ll_sorted_contains(lst, 4));
    // 961 = 31^2
    CuAssertIntEquals(tc, 0, ll_sorted_contains(lst, 960));
    CuAssertIntEquals(tc, 1, ll_sorted_contains(lst, 961));
    CuAssertIntEquals(tc, 0, ll_sorted_contains(lst, 962));
    // 1024 = 32^2
    CuAssertIntEquals(tc, 0, ll_sorted_contains(lst, 1023));
    CuAssertIntEquals(tc, 1, ll_sorted_contains(lst, 1024));
    CuAssertIntEquals(tc, 0, ll_sorted_contains(lst, 1025));
}

void test_ll_sorted_index_of_1(CuTest* tc) {
    int i;
    ll* lst = ll_new(32);
    // Fill the list with L[i] = i^2
    for (i=0; i<=10000; i++)
        ll_append(lst, i*i);
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, -1));
    CuAssertIntEquals(tc, 0, ll_sorted_index_of(lst, 0));
    CuAssertIntEquals(tc, 1, ll_sorted_index_of(lst, 1));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 2));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 3));
    CuAssertIntEquals(tc, 2, ll_sorted_index_of(lst, 4));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 5));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 960));
    CuAssertIntEquals(tc, 31, ll_sorted_index_of(lst, 961));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 962));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 1023));
    CuAssertIntEquals(tc, 32, ll_sorted_index_of(lst, 1024));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 1025));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 9999));
    CuAssertIntEquals(tc, 100, ll_sorted_index_of(lst, 10000));
    CuAssertIntEquals(tc, -1, ll_sorted_index_of(lst, 10001));
    CuAssertIntEquals(tc, -1,    ll_sorted_index_of(lst, 10000 * 10000 - 1));
    CuAssertIntEquals(tc, 10000, ll_sorted_index_of(lst, 10000 * 10000    ));
    CuAssertIntEquals(tc, -1,    ll_sorted_index_of(lst, 10000 * 10000 + 1));
    // test again (jump accessors)
    CuAssertIntEquals(tc, -1,    ll_sorted_index_of(lst, 10000 * 10000 - 1));
    CuAssertIntEquals(tc, 10000, ll_sorted_index_of(lst, 10000 * 10000    ));
    CuAssertIntEquals(tc, -1,    ll_sorted_index_of(lst, 10000 * 10000 + 1));
}

void test_il_remove_index_range_1(CuTest* tc) {
    il* lst = il_new(4);
    il_append(lst, 0);
    il_append(lst, 1);
    il_append(lst, 2);
    il_append(lst, 3);
    il_append(lst, 4);
    il_append(lst, 5);
    il_remove_index_range(lst, 2, 2);
    // [0 1 2 3 4 5] -> [0 1 4 5]
    CuAssertIntEquals(tc, 4, il_size(lst));
    CuAssertIntEquals(tc, 4, il_get(lst, 2));
}

void test_sl_split_1(CuTest* tc) {
    sl* s = sl_split(NULL, "hello world this is a test", " ");
    CuAssertPtrNotNull(tc, s);
    CuAssertIntEquals(tc, 6, sl_size(s));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 0), "hello"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 1), "world"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 2), "this"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 3), "is"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 4), "a"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 5), "test"));
    sl_free2(s);
}

void test_sl_split_2(CuTest* tc) {
    int i;
    sl* s = sl_split(NULL, "hello  world  this  is  a  test     ", "  ");
    CuAssertPtrNotNull(tc, s);
    printf("got: ");
    for (i=0; i<sl_size(s); i++)
        printf("/%s/ ", sl_get(s, i));
    printf("\n");
    CuAssertIntEquals(tc, 8, sl_size(s));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 0), "hello"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 1), "world"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 2), "this"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 3), "is"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 4), "a"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 5), "test"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 6), ""));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 7), " "));
    sl_free2(s);
}

void test_sl_split_3(CuTest* tc) {
    sl* s, *s2;
    s = sl_new(1);
    sl_append(s, "guard");
    s2 = sl_split(s, "XYhelloXYworldXYXY", "XY");
    CuAssertPtrNotNull(tc, s2);
    CuAssertPtrEquals(tc, s, s2);
    CuAssertIntEquals(tc, 5, sl_size(s));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 0), "guard"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 1), ""));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 2), "hello"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 3), "world"));
    CuAssertIntEquals(tc, 0, strcmp(sl_get(s, 4), ""));
    sl_free2(s);
}

void test_il_new(CuTest* tc) {
    il* x = NULL;
    x = il_new(10);
    CuAssert(tc, "new", x != NULL);
    CuAssertIntEquals(tc, il_size(x), 0);
    CuAssertPtrEquals(tc, x->head, NULL);
    CuAssertPtrEquals(tc, x->tail, NULL);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, il_check_sorted_ascending(x, 0), 0);
    il_free(x);
}

void test_il_size(CuTest* tc) {
    il* x = il_new(10);
    int i, N = 100;
    for (i=0; i<N; i++) {
        il_push(x, i);
        CuAssertIntEquals(tc, i+1, il_size(x));
    }
    il_free(x);
}

void test_il_get_push(CuTest* tc) {
    il* x = il_new(10);
    il_push(x,10);
    CuAssertIntEquals(tc, 10, il_get(x, 0));
    il_push(x,20);
    CuAssertIntEquals(tc, 10, il_get(x, 0));
    CuAssertIntEquals(tc, 20, il_get(x, 1));
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    il_free(x);
}

void test_il_push_pop(CuTest* tc) {
    il* x = il_new(10);
    il_push(x,10);
    CuAssertIntEquals(tc, 10, il_get(x, 0));
    il_push(x,20);
    CuAssertIntEquals(tc, 10, il_get(x, 0));
    CuAssertIntEquals(tc, 20, il_get(x, 1));
    CuAssertIntEquals(tc, 20, il_pop(x));
    CuAssertIntEquals(tc, 10, il_pop(x));
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    il_free(x);
}

void test_il_push_pop2(CuTest* tc) {
    int i, N=100;
    il* x = il_new(10);
    for (i=0; i<N; i++)
        il_push(x,i);
    for (i=0; i<N; i++)
        CuAssertIntEquals(tc, N-i-1, il_pop(x));
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    il_free(x);
}


void test_il(CuTest* tc) {
    il* x = il_new(10);
    il_push(x,10);
    CuAssertIntEquals(tc, 10, il_get(x, 0));
    il_push(x,20);
    CuAssertIntEquals(tc, 10, il_get(x, 0));
    CuAssertIntEquals(tc, 20, il_get(x, 1));
    il_free(x);
}

void test_il_remove_value(CuTest* tc) {
    il* x = il_new(5);
    il_push(x,10);
    il_push(x,20);
    il_push(x,30);
    il_push(x,87);
    il_push(x,87);
    il_push(x,87);
    il_push(x,87);
    il_push(x,87);
    il_push(x,92);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, 8, il_remove_value(x, 92));
    CuAssertIntEquals(tc, -1, il_remove_value(x, 37));
    CuAssertIntEquals(tc, -1, il_remove_value(x, 0));
    CuAssertIntEquals(tc, 2, il_remove_value(x, 30));
    CuAssertIntEquals(tc, 0, il_remove_value(x, 10));
    CuAssertIntEquals(tc, 0, il_remove_value(x, 20));
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    il_free(x);
}

void test_il_contains(CuTest* tc) {
    il* x = il_new(4);
    il_push(x,10);
    il_push(x,20);
    il_push(x,30);
    il_push(x,30);
    il_push(x,30);
    il_push(x,41);
    il_push(x,30);
    il_push(x,81);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, 1, il_contains(x, 10));
    CuAssertIntEquals(tc, 1, il_contains(x, 20));
    CuAssertIntEquals(tc, 1, il_contains(x, 30));
    CuAssertIntEquals(tc, 1, il_contains(x, 81));
    CuAssertIntEquals(tc, 1, il_contains(x, 41));
    CuAssertIntEquals(tc, 0, il_contains(x, 42));
    il_remove_value(x, 41);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, 0, il_contains(x, 41));
    il_free(x);
}

void test_il_insert_ascending(CuTest* tc) {
    int i;
    il* x = il_new(4);
    il_insert_ascending(x,2);
    il_insert_ascending(x,4);
    il_insert_ascending(x,8);
    il_insert_ascending(x,5);
    il_insert_ascending(x,6);
    il_insert_ascending(x,7);
    il_insert_ascending(x,1);
    il_insert_ascending(x,3);
    il_insert_ascending(x,0);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, il_check_sorted_ascending(x, 0), 0);
    for (i=0;i<il_size(x);i++)
        CuAssertIntEquals(tc, i, il_get(x, i));
    for (i=0;i<il_size(x);i++)
        CuAssertIntEquals(tc, i, il_find_index_ascending(x, i));
    il_free(x);
}

void test_il_insert_descending(CuTest* tc) {
    int i;
    il* x = il_new(4);
    il_insert_descending(x,2);
    il_insert_descending(x,4);
    il_insert_descending(x,8);
    il_insert_descending(x,5);
    il_insert_descending(x,6);
    il_insert_descending(x,7);
    il_insert_descending(x,1);
    il_insert_descending(x,3);
    il_insert_descending(x,0);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, il_check_sorted_descending(x, 0), 0);
    for (i=0;i<il_size(x);i++)
        CuAssertIntEquals(tc, il_size(x)-i-1, il_get(x, i));
    il_free(x);
}


void test_il_insert_unique_ascending(CuTest* tc) {
    int i;
    il* x = il_new(4);
    il_insert_unique_ascending(x,2);
    il_insert_unique_ascending(x,4);
    il_insert_unique_ascending(x,4);
    il_insert_unique_ascending(x,7);
    il_insert_unique_ascending(x,4);
    il_insert_unique_ascending(x,4);
    il_insert_unique_ascending(x,8);
    il_insert_unique_ascending(x,5);
    il_insert_unique_ascending(x,0);
    il_insert_unique_ascending(x,5);
    il_insert_unique_ascending(x,5);
    il_insert_unique_ascending(x,5);
    il_insert_unique_ascending(x,4);
    il_insert_unique_ascending(x,5);
    il_insert_unique_ascending(x,6);
    il_insert_unique_ascending(x,7);
    il_insert_unique_ascending(x,7);
    il_insert_unique_ascending(x,7);
    il_insert_unique_ascending(x,7);
    il_insert_unique_ascending(x,1);
    il_insert_unique_ascending(x,1);
    il_insert_unique_ascending(x,3);
    il_insert_unique_ascending(x,1);
    il_insert_unique_ascending(x,1);
    il_insert_unique_ascending(x,0);
    il_print(x);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    CuAssertIntEquals(tc, il_check_sorted_ascending(x, 1), 0);
    for (i=0;i<il_size(x);i++)
        CuAssertIntEquals(tc, i, il_get(x, i));
    il_free(x);
}

void test_il_copy(CuTest* tc) {
    int i, N=60, start=10, length=10;
    int buf[N];
    il* x = il_new(4);
    memset(buf, 0, N);
    for (i=0;i<N;i++) 
        il_push(x,i);
    il_copy(x, start, length, buf);
    for (i=0;i<length;i++) 
        CuAssertIntEquals(tc, start+i, buf[i]);
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    il_free(x);
}

void test_il_dupe(CuTest* tc) {
    int i, N=63;
    il* x = il_new(4), *y;
    for (i=0;i<N;i++) 
        il_push(x,i);
    y = il_dupe(x);
    for (i=0;i<N;i++) 
        CuAssertIntEquals(tc, i, il_get(y, i));
    for (i=0;i<N;i++) 
        il_pop(x);
    CuAssertIntEquals(tc, N, il_size(y));
    CuAssertIntEquals(tc, il_check_consistency(x), 0);
    il_free(x);
    il_free(y);
}

void test_delete(CuTest* tc) {
    il* bl = il_new(4);
    il_push(bl, 42);
    il_push(bl, 43);
    il_push(bl, 47);
    il_push(bl, 49);

    il_remove(bl, 0);
    il_remove(bl, 0);
    il_remove(bl, 0);
    il_remove(bl, 0);

    CuAssertIntEquals(tc, il_size(bl), 0);
    CuAssertPtrEquals(tc, bl->head, NULL);
    CuAssertPtrEquals(tc, bl->tail, NULL);
    CuAssertIntEquals(tc, il_check_consistency(bl), 0);
    il_free(bl);
}

void test_delete_2(CuTest* tc) {
    il* bl = il_new(2);
    il_push(bl, 42);
    il_push(bl, 43);
    il_push(bl, 47);
    il_push(bl, 49);

    il_remove(bl, 0);
    il_remove(bl, 0);
    il_remove(bl, 0);
    il_remove(bl, 0);

    CuAssertIntEquals(tc, il_size(bl), 0);
    CuAssertPtrEquals(tc, bl->head, NULL);
    CuAssertPtrEquals(tc, bl->tail, NULL);
    CuAssertIntEquals(tc, il_check_consistency(bl), 0);
    il_free(bl);
}

void test_delete_3(CuTest* tc) {
    il* bl;
    bl = il_new(2);
    il_push(bl, 42);
    il_push(bl, 43);
    il_push(bl, 47);
    il_push(bl, 49);

    il_remove(bl, 3);
    il_remove(bl, 2);
    il_remove(bl, 1);
    il_remove(bl, 0);

    CuAssertIntEquals(tc, il_size(bl), 0);
    CuAssertPtrEquals(tc, bl->head, NULL);
    CuAssertPtrEquals(tc, bl->tail, NULL);
    CuAssertIntEquals(tc, il_check_consistency(bl), 0);
    il_free(bl);
}

void test_set(CuTest* tc) {
    il* bl;
    bl = il_new(2);
    CuAssertIntEquals(tc, il_size(bl), 0);
    il_push(bl, 42);
    il_push(bl, 43);
    il_push(bl, 47);
    il_push(bl, 49);

    il_set(bl, 0, 0);
    il_set(bl, 1, 1);
    il_set(bl, 2, 2);

    CuAssertIntEquals(tc, il_size(bl), 4);
    CuAssertIntEquals(tc, 0, il_get(bl, 0));
    CuAssertIntEquals(tc, 1, il_get(bl, 1));
    CuAssertIntEquals(tc, 2, il_get(bl, 2));
    CuAssertIntEquals(tc, il_check_consistency(bl), 0);
    il_free(bl);
}

void test_delete_4(CuTest* tc) {
    int i, j, N;
    il* bl = il_new(20);
    N = 100;
    for (i=0; i<N; i++)
        il_push(bl, i);

    for (i=0; i<N; i++) {
        int ind = rand() % il_size(bl);
        il_remove(bl, ind);

        for (j=1; j<il_size(bl); j++) {
            CuAssert(tc, "mono", (il_get(bl, j) - il_get(bl, j-1)) > 0);
        }
    }

    CuAssertIntEquals(tc, il_size(bl), 0);
    CuAssertPtrEquals(tc, bl->head, NULL);
    CuAssertPtrEquals(tc, bl->tail, NULL);
    CuAssertIntEquals(tc, il_check_consistency(bl), 0);
    il_free(bl);
}

/******************************************************************************/
/****************************** double lists **********************************/
/******************************************************************************/


void test_dl_push(CuTest* tc) {
    dl* bl;
    bl = dl_new(2);
    CuAssertIntEquals(tc, il_size(bl), 0);
    dl_push(bl, 42.0);
    dl_push(bl, 43.0);
    dl_push(bl, 47.0);
    dl_push(bl, 49.0);

    dl_set(bl, 0, 0.0);
    dl_set(bl, 1, 1.0);
    dl_set(bl, 2, 2.0);

    CuAssertIntEquals(tc, il_size(bl), 4);
    CuAssert(tc, "dl", 0.0 == dl_get(bl, 0));
    CuAssert(tc, "dl", 1.0 == dl_get(bl, 1));
    CuAssert(tc, "dl", 2.0 == dl_get(bl, 2));
    CuAssertIntEquals(tc, dl_check_consistency(bl), 0);
    dl_free(bl);
}

void test_bl_extend(CuTest *tc) {
    bl* list = bl_new(10, sizeof(int));
    CuAssertIntEquals(tc, bl_size(list), 0);
    int *new1 = bl_extend(list);
    CuAssertPtrNotNull(tc, new1);
    CuAssertIntEquals(tc, bl_size(list), 1);
    *new1 = 10;
    int *new2 = bl_access(list, 0);
    CuAssertPtrEquals(tc, new2, new1);
    bl_free(list);
}

///////////

static void addsome(sl* s, const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    sl_appendvf(s, fmt, va);
    va_end(va);
}

void test_sl_join(CuTest* tc) {
    char* s1;
    sl* s = sl_new(4);
    sl_append(s, "123");
    sl_appendf(s, "%1$s%1$s", "testing");
    addsome(s, "%i", 456);
    sl_insert(s, 1, "inserted");
    sl_insertf(s, 2, "%s%s", "ins", "ertedf");
    s1 = sl_join(s, "");
    CuAssertStrEquals(tc, "123insertedinsertedftestingtesting456", s1);
    free(s1);
    s1 = sl_join(s, "--");
    CuAssertStrEquals(tc, "123--inserted--insertedf--testingtesting--456", s1);
    free(s1);
    s1 = sl_join_reverse(s, "--");
    CuAssertStrEquals(tc, "456--testingtesting--insertedf--inserted--123", s1);
    free(s1);

    sl_free2(s);
}

