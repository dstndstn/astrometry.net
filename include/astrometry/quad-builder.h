/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#ifndef QUAD_BUILDER_H
#define QUAD_BUILDER_H

#include "astrometry/an-bool.h"

struct potential_quad {
    double midAB[3];
    double Ax, Ay;
    double costheta, sintheta;
    int iA, iB;
    int staridA, staridB;
    int* inbox;
    int ninbox;
    anbool scale_ok;

    // user-defined check passed?
    anbool check_ok;
};
typedef struct potential_quad pquad_t;

struct quadbuilder;
typedef struct quadbuilder quadbuilder_t;

struct quadbuilder {
    // FIXME -- could replace this with a function to get xyz for a given index
    // (eg, using starkd)
    double* starxyz;

    // Indices into a (presumed) larger array.  Ie, it's expected that you're
    // cutting down to a subset of the stars, in "starxyz", whose indices are in
    // "starinds", and the number of which is "Nstars".
    // (thus "starinds" has length "Nstars" but its contents can be > Nstars)
    int* starinds;
    int Nstars;
    int dimquads;

    // scale limits, in diameter-squared on the unit sphere
    double quadd2_low;
    double quadd2_high;
    // enable scale checks?
    anbool check_scale_low;
    anbool check_scale_high;

    // FIXME -- could add a method to find potential B stars given an A star.
    // (eg, allquads)

    // called to check whether a choice of stars A,B is acceptable.
    anbool (*check_AB_stars)(quadbuilder_t* qb, pquad_t* pq, void* token);
    void* check_AB_stars_token;

    // called when the third, fourth, ... stars are added.
    anbool (*check_partial_quad)(quadbuilder_t* qb, unsigned int* quad, int nstars, void* token);
    void* check_partial_quad_token;

    // called when all the stars have been added.
    anbool (*check_full_quad)(quadbuilder_t* qb, unsigned int* quad, int nstars, void* token);
    void* check_full_quad_token;

    // called to decide which of the given stars are accetable.
    // must compact the acceptable star indices into the bottom of the "sC" array and 
    // return the new number of acceptable stars.
    int (*check_internal_stars)(quadbuilder_t* qb, int sA, int sB, int* sC, int NC, void* token);
    void* check_internal_stars_token;

    // called when an acceptable quad has been found.
    void (*add_quad)(quadbuilder_t* qb, unsigned int* stars, void* token);
    void* add_quad_token;


    // set this to stop a qb_create() call.
    anbool stop_creating;

    //
    int nbadscale;

    // internal:
    int Ncq;
    void* pquads;
    //pquad_t* pquads;
    int* inbox;
};

int quadbuilder_create(quadbuilder_t* qb);

quadbuilder_t* quadbuilder_init();

void quadbuilder_free(quadbuilder_t* qb);

//void quadbuilder_free_static();



#endif
