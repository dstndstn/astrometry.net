/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef QUAD_UTILS_H
#define QUAD_UTILS_H

#include "astrometry/starkd.h"
#include "astrometry/codefile.h"
#include "astrometry/quadfile.h"
#include "astrometry/an-bool.h"

void quad_compute_star_code(const double* starxyz, double* code, int dimquads);

void quad_flip_parity(const double* code, double* flipcode, int dimcode);

int quad_compute_code(const unsigned int* quad, int dimquads, startree_t* starkd, 
                      double* code);

void quad_enforce_invariants(unsigned int* quad, double* code,
                             int dimquads, int dimcodes);

anbool quad_obeys_invariants(unsigned int* quad, double* code,
                             int dimquads, int dimcodes);

#endif
