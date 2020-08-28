/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stddef.h>

#include "fitsbin.h"
#include "fitsioutils.h"
#include "codefile.h"
#include "ioutils.h"
#include "errors.h"

#include "cutest.h"




void test_codefile_big(CuTest* ct) {
    int i, d;
    int D = 4;
    int N = 250000000;
    double code[4];
    double ecode[4];
    codefile_t* cf;
    char* fn;

    fits_use_error_system();

    fn = create_temp_file("test_codefile", NULL);

    cf = codefile_open_for_writing(fn);
    CuAssertPtrNotNull(ct, cf);

    CuAssertIntEquals(ct, 0, codefile_write_header(cf));

    for (i=0; i<N; i++) {
        for (d=0; d<D; d++) {
            code[d] = i+d;
        }
        if (i % 1000000 == 0)
            printf("Writing %i\n", i);
        if (codefile_write_code(cf, code)) {
            ERROR("Failed to write code %i", i);
            exit(-1);
        }
    }

    if (codefile_fix_header(cf)) {
        ERROR("failed to fix header");
        exit(-1);
    }

    if (codefile_close(cf)) {
        ERROR("Failed to close cf\n");
    }
    printf("Wrote %s\n", fn);

    cf = codefile_open(fn);
    CuAssertPtrNotNull(ct, cf);

    for (i=0; i<N; i++) {
        for (d=0; d<D; d++) {
            ecode[d] = i+d;
        }
        codefile_get_code(cf, i, code);
        if (memcmp(ecode, code, sizeof(double) * 4)) {
            ERROR("Failed on code %i: expected %f,%f,%f,%f, got %f,%f,%f,%f\n",
                  i, ecode[0], ecode[1], ecode[2], ecode[3],
                  code[0], code[1], code[2], code[3]);
            exit(-1);
        }
    }

    if (codefile_close(cf)) {
        ERROR("Failed to close cf\n");
    }

}
