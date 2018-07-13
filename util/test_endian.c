/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdint.h>
#include <stdio.h>

#include "an-endian.h"
#include "cutest.h"

// Embarassing, but apparently necessary, since I fscked it up once.


void test_endian(CuTest* ct) {
    char test64[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    uint64_t lbe = 0x0102030405060708LL;
    //uint64_t lle = 0x0807060504030201LL;
    uint64_t* ltest = (uint64_t*)test64;
    uint64_t lval;

    char test32[4] = { 1, 2, 3, 4 };
    uint32_t ibe = 0x01020304;
    uint32_t ile = 0x04030201;
    uint32_t* itest = (uint32_t*)test32;
    uint32_t ival;

    char test16[2] = { 1, 2 };
    uint16_t sbe = 0x0102;
    uint16_t sle = 0x0201;
    uint16_t* stest = (uint16_t*)test16;
    uint16_t sval;

#if IS_BIG_ENDIAN
    printf("Big endian\n");
    ival = *itest;
    CuAssertIntEquals(ct, ival, ibe);
#else
    printf("Little endian\n");
    ival = *itest;
    CuAssertIntEquals(ct, ival, ile);
#endif

    // 64-bit

    lval = *ltest;
    v64_hton(&lval);
    CuAssert(ct, "lval == lbe", lval == lbe);

    lval = *ltest;
    v64_ntoh(&lval);
    CuAssert(ct, "lval == lbe", lval == lbe);

    /*
     lval = *ltest;
     v64_htole(&ival);
     CuAssert(ct, "lval == lle", lval == lle);

     lval = *ltest;
     v64_letoh(&lval);
     CuAssert(ct, "lval == lle", lval == lle);
     */

    // 32-bit

    ival = *itest;
    v32_hton(&ival);
    CuAssertIntEquals(ct, ival, ibe);

    ival = *itest;
    v32_ntoh(&ival);
    CuAssertIntEquals(ct, ival, ibe);

    ival = *itest;
    v32_htole(&ival);
    CuAssertIntEquals(ct, ival, ile);

    ival = *itest;
    v32_letoh(&ival);
    CuAssertIntEquals(ct, ival, ile);

    // 16-bit

    sval = *stest;
    v16_hton(&sval);
    CuAssertIntEquals(ct, sval, sbe);

    sval = *stest;
    v16_ntoh(&sval);
    CuAssertIntEquals(ct, sval, sbe);

    sval = *stest;
    v16_htole(&sval);
    CuAssertIntEquals(ct, sval, sle);

    /*
     sval = *stest;
     v16_letoh(&sval);
     CuAssertIntEquals(ct, sval, sle);
     */
}

