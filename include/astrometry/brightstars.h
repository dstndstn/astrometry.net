/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef BRIGHTSTARS_H
#define BRIGHTSTARS_H

struct brightstar {
    // Don't change the order of these fields - the included datafile depends on this order!
    char* name;
    char* common_name;
    double ra;
    double dec;
    double Vmag;
};
typedef struct brightstar brightstar_t;

int bright_stars_n();
const brightstar_t* bright_stars_get(int starindex);

#endif
