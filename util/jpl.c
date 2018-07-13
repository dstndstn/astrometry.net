/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>

#include "bl.h"

struct orbital_elements {
    double a, e, I, Omega, pomega, M;
    double mjd;
};
typedef struct orbital_elements orbital_elements_t;


bl* jpl_parse_orbital_elements(const char* str, bl* lst) {
    double jd;
    double a, e, I, Omega, pomega, M;
    double QR, Tp, N, TA, AD, PR;
    char adstr[5];
    char datestr[12];
    char timestr[14];
    char tzstr[5];
    sscanf(s, "%lf = %4s %11s %13s %4s EC= %lf QR= %lf IN=%lf"
           " OM= %lf W = %lf Tp= %lf N = %lf MA= %lf TA= %lf"
           " A = %lf AD= %lf PR= %lf",
           &jd, adstr, datestr, timestr, tzstr, &e, &QR, &I,
           &Omega, &pomega, &Tp, &N, &M, &TA, &a, &AD, &PR);
}

