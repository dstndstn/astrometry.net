/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>

#include "quad-utils.h"
#include "starutil.h"
#include "codefile.h"
#include "starkd.h"
#include "errors.h"
#include "log.h"

void quad_compute_star_code(const double* starxyz, double* code, int dimquads) {
    double Ax=0, Ay=0;
    double Bx=0, By=0;
    double ABx, ABy;
    double scale, invscale;
    double costheta, sintheta;
    double midAB[3];
    Unused anbool ok;
    int i;
    const double *sA, *sB;

    sA = starxyz;
    sB = starxyz + 3;
    star_midpoint(midAB, sA, sB);
    ok = star_coords(sA, midAB, TRUE, &Ay, &Ax);
    assert(ok);
    ok = star_coords(sB, midAB, TRUE, &By, &Bx);
    assert(ok);
    ABx = Bx - Ax;
    ABy = By - Ay;
    scale = (ABx * ABx) + (ABy * ABy);
    invscale = 1.0 / scale;
    costheta = (ABy + ABx) * invscale;
    sintheta = (ABy - ABx) * invscale;

    for (i=2; i<dimquads; i++) {
        const double* starpos;
        double Dx=0, Dy=0;
        double ADx, ADy;
        double x, y;
        starpos = starxyz + 3*i;
        ok = star_coords(starpos, midAB, TRUE, &Dy, &Dx);
        assert(ok);
        ADx = Dx - Ax;
        ADy = Dy - Ay;
        x =  ADx * costheta + ADy * sintheta;
        y = -ADx * sintheta + ADy * costheta;
        code[2*(i-2)+0] = x;
        code[2*(i-2)+1] = y;
    }
}

void quad_flip_parity(const double* code, double* flipcode, int dimcode) {
    int i;
    // swap CX <-> CY, DX <-> DY.
    for (i=0; i<dimcode/2; i++) {
        // use tmp in code "code" == "flipcode"
        double tmp;
        tmp = code[2*i+1];
        flipcode[2*i+1] = code[2*i+0];
        flipcode[2*i+0] = tmp;
    }
}

int quad_compute_code(const unsigned int* quad, int dimquads, startree_t* starkd, 
                      double* code) {
    int i;
    double starxyz[3 * DQMAX];
    for (i=0; i<dimquads; i++) {
        if (startree_get(starkd, quad[i], starxyz + 3*i)) {
            ERROR("Failed to get stars belonging to a quad.\n");
            return -1;
        }
    }
    quad_compute_star_code(starxyz, code, dimquads);
    return 0;
}

anbool quad_obeys_invariants(unsigned int* quad, double* code,
                             int dimquads, int dimcodes) {
    double sum;
    int i;
    // check the invariant that (cx + dx + ...) / (dimquads-2) <= 1/2
    sum = 0.0;
    for (i=0; i<(dimquads-2); i++)
        sum += code[2*i];
    sum /= (dimquads-2);
    if (sum > 0.5)
        return FALSE;

    // check the invariant that cx <= dx <= ....
    for (i=0; i<(dimquads-3); i++)
        if (code[2*i] > code[2*(i+1)])
            return FALSE;
    return TRUE;
}

void quad_enforce_invariants(unsigned int* quad, double* code,
                             int dimquads, int dimcodes) {
    double sum;
    int i;

    // here we add the invariant that (cx + dx + ...) / (dimquads-2) <= 1/2
    sum = 0.0;
    for (i=0; i<dimcodes/2; i++)
        sum += code[2*i];
    sum /= (dimcodes/2);
    if (sum > 0.5) {
        logdebug("Flipping code to ensure mean(x)<=0.5\n");
        // swap the labels of A,B.
        int tmp = quad[0];
        quad[0] = quad[1];
        quad[1] = tmp;
        // rotate the code 180 degrees.
        for (i=0; i<dimcodes; i++)
            code[i] = 1.0 - code[i];
    }

    // here we add the invariant that cx <= dx <= ....
    for (i=0; i<(dimquads-2); i++) {
        int j;
        int jsmallest;
        double smallest;
        double x1;
        double dtmp;
        int tmp;

        x1 = code[2*i];
        jsmallest = -1;
        smallest = x1;
        for (j=i+1; j<(dimquads-2); j++) {
            double x2 = code[2*j];
            if (x2 < smallest) {
                smallest = x2;
                jsmallest = j;
            }
        }
        if (jsmallest == -1)
            continue;
        j = jsmallest;
        // swap the labels.
        tmp = quad[i+2];
        quad[i+2] = quad[j+2];
        quad[j+2] = tmp;
        // swap the code values.
        dtmp = code[2*i];
        code[2*i] = code[2*j];
        code[2*j] = dtmp;
        dtmp = code[2*i+1];
        code[2*i+1] = code[2*j+1];
        code[2*j+1] = dtmp;
    }
}

