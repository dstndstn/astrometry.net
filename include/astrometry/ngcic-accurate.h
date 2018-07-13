/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef NGCIC_ACCURATE_H
#define NGCIC_ACCURATE_H

/*
 The accurate NGC/IC positions database can be found here:
 http://www.ngcic.org/corwin/default.htm
 */

struct ngcic_accurate {
    // true: NGC.  false: IC.
    int is_ngc;
    // NGC/IC number
    int id;
    // RA,Dec in B2000.0 degrees
    float ra;
    float dec;
};
typedef struct ngcic_accurate ngcic_accurate;

int ngcic_accurate_get_radec(int is_ngc, int id, float* ra, float* dec);

int ngcic_accurate_num_entries();

ngcic_accurate* ngcic_accurate_get_entry(int i);

#endif
