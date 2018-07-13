/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>

#include "ngcic-accurate.h"
#include "an-bool.h"

static ngcic_accurate ngcic_acc[] = {
#include "ngcic-accurate-entries.c"
};

int ngcic_accurate_get_radec(int is_ngc, int id, float* ra, float* dec) {
    int i, N;
    N = sizeof(ngcic_acc) / sizeof(ngcic_accurate);
    for (i=0; i<N; i++) {
        if ((ngcic_acc[i].is_ngc != is_ngc) ||
            (ngcic_acc[i].id != id))
            continue;
        *ra = ngcic_acc[i].ra;
        *dec = ngcic_acc[i].dec;
        return 0;
    }
    return -1;
}

int ngcic_accurate_num_entries() {
    return sizeof(ngcic_acc) / sizeof(ngcic_accurate);
}

ngcic_accurate* ngcic_accurate_get_entry(int i) {
    int N = sizeof(ngcic_acc) / sizeof(ngcic_accurate);
    if (i < 0 || i >= N)
        return NULL;
    return ngcic_acc + i;
}
