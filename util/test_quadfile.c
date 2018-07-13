/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stddef.h>

#include "fitsbin.h"
#include "fitsioutils.h"
#include "quadfile.h"
#include "errors.h"

#include "cutest.h"

void test_quadfile_inmemory_big(CuTest* ct) {
    int i, d;
    int D = 4;
    int N = 1000000000;
    unsigned int quad[4];
    unsigned int equad[4];
    quadfile_t* qf;

    qf = quadfile_open_in_memory();
    CuAssertPtrNotNull(ct, qf);

    CuAssertIntEquals(ct, 0, quadfile_write_header(qf));

    for (i=0; i<N; i++) {
        for (d=0; d<D; d++) {
            quad[d] = i+d;
        }
        if (i % 1000000 == 0)
            printf("Writing %i\n", i);
        if (quadfile_write_quad(qf, quad)) {
            ERROR("Failed to write quad %i", i);
            exit(-1);
        }
    }

    if (quadfile_fix_header(qf)) {
        ERROR("failed to fix header");
        exit(-1);
    }

    printf("Switching to reading...\n");
    if (quadfile_switch_to_reading(qf)) {
        ERROR("Failed to switch to reading");
        exit(-1);
    }

    printf("Reading...\n");
    for (i=0; i<N; i++) {
        for (d=0; d<D; d++) {
            equad[d] = i+d;
        }
        if (i % 1000000 == 0)
            printf("Reading %i\n", i);
        if (quadfile_get_stars(qf, i, quad)) {
            ERROR("Failed to read %i\n", i);
            exit(-1);
        }
        if (memcmp(equad, quad, sizeof(unsigned int) * 4)) {
            ERROR("Failed on quad %i: expected %i,%i,%i,%i, got %i,%i,%i,%i\n",
                  i, equad[0], equad[1], equad[2], equad[3],
                  quad[0], quad[1], quad[2], quad[3]);
            exit(-1);
        }
    }

    if (quadfile_close(qf)) {
        ERROR("Failed to close qf\n");
    }
}
