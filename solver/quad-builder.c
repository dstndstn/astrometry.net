/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>

#include "quad-builder.h"
#include "quad-utils.h"
#include "mathutil.h"
#include "errors.h"
#include "log.h"

struct quad {
    unsigned int star[DQMAX];
};
typedef struct quad quad;

static void check_scale(quadbuilder_t* qb, pquad_t* pq) {
    double *sA, *sB;
    double s2;
    double Bx=0, By=0;
    double invscale;
    double ABx, ABy;
    Unused anbool ok;
    if (!(qb->check_scale_low || qb->check_scale_high)) {
        logverb("Not checking scale\n");
        // ??
        pq->scale_ok = TRUE;
        return;
    }
    sA = qb->starxyz + pq->iA * 3;
    sB = qb->starxyz + pq->iB * 3;
    // s2: squared AB dist
    s2 = distsq(sA, sB, 3);
    pq->scale_ok = TRUE;
    if (qb->check_scale_low && s2 < qb->quadd2_low)
        pq->scale_ok = FALSE;
    if (pq->scale_ok && qb->check_scale_high && s2 > qb->quadd2_high)
        pq->scale_ok = FALSE;
    if (!pq->scale_ok) {
        qb->nbadscale++;
        return;
    }
    star_midpoint(pq->midAB, sA, sB);
    pq->scale_ok = TRUE;
    pq->staridA = qb->starinds[pq->iA];
    pq->staridB = qb->starinds[pq->iB];
    ok = star_coords(sA, pq->midAB, TRUE, &pq->Ay, &pq->Ax);
    assert(ok);
    ok = star_coords(sB, pq->midAB, TRUE, &By, &Bx);
    assert(ok);
    ABx = Bx - pq->Ax;
    ABy = By - pq->Ay;
    invscale = 1.0 / (ABx*ABx + ABy*ABy);
    pq->costheta = (ABy + ABx) * invscale;
    pq->sintheta = (ABy - ABx) * invscale;
    //nabok++;
}

static int
check_inbox(pquad_t* pq, int* inds, int ninds, double* stars) {
    int i, ind;
    double* starpos;
    double Dx=0, Dy=0;
    double ADx, ADy;
    double x, y;
    int destind = 0;
    anbool ok;
    for (i=0; i<ninds; i++) {
        double r;
        ind = inds[i];
        starpos = stars + ind*3;
        logverb("Star position: [%.5f, %.5f, %.5f]\n",
                starpos[0], starpos[1], starpos[2]);
        logverb("MidAB: [%.5f, %.5f, %.5f]\n",
                pq->midAB[0], pq->midAB[1], pq->midAB[2]);

        ok = star_coords(starpos, pq->midAB, TRUE, &Dy, &Dx);
        if (!ok) {
            logverb("star coords not ok\n");
            continue;
        }
        ADx = Dx - pq->Ax;
        ADy = Dy - pq->Ay;
        x =  ADx * pq->costheta + ADy * pq->sintheta;
        y = -ADx * pq->sintheta + ADy * pq->costheta;
        // make sure it's in the circle centered at (0.5, 0.5)...
        // (x-1/2)^2 + (y-1/2)^2   <=   r^2
        // x^2-x+1/4 + y^2-y+1/4   <=   (1/sqrt(2))^2
        // x^2-x + y^2-y + 1/2     <=   1/2
        // x^2-x + y^2-y           <=   0
        r = (x*x - x) + (y*y - y);
        if (r > 0.0) {
            logverb("star not in circle\n");
            continue;
        }
        inds[destind] = ind;
        destind++;
    }
    return destind;
}

/**
 inbox, ninbox: the stars we have to work with.
 starinds: the star identifiers (indexed by the contents of 'inbox')
 - ie, starinds[inbox[0]] is an externally-recognized star identifier.
 q: where we record the star identifiers
 starnum: which star we're adding: eg, A=0, B=1, C=2, ... dimquads-1.
 beginning: the first index in "inbox" to assign to star 'starnum'.
 */
static void add_interior_stars(quadbuilder_t* qb,
                               int ninbox, int* inbox, quad* q,
                               int starnum, int dimquads, int beginning) {
    int i;
    for (i=beginning; i<ninbox; i++) {
        int iC = inbox[i];
        q->star[starnum] = qb->starinds[iC];
        // Did we just add the last star?
        if (starnum >= dimquads-1) {
            if (qb->check_full_quad &&
                !qb->check_full_quad(qb, q->star, dimquads, qb->check_full_quad_token))
                continue;

            qb->add_quad(qb, q->star, qb->add_quad_token);
        } else {
            if (qb->check_partial_quad &&
                !qb->check_partial_quad(qb, q->star, starnum+1, qb->check_partial_quad_token))
                continue;
            // Recurse.
            add_interior_stars(qb, ninbox, inbox, q, starnum+1, dimquads, i+1);
        }
        if (qb->stop_creating)
            return;
    }
}

quadbuilder_t* quadbuilder_init() {
    quadbuilder_t* qb = calloc(1, sizeof(quadbuilder_t));
    return qb;
}

void quadbuilder_free(quadbuilder_t* qb) {
    free(qb->inbox);
    free(qb->pquads);
    free(qb);
}

int quadbuilder_create(quadbuilder_t* qb) {
    int iA=0, iB, iC, iD, newpoint;
    int ninbox;
    int i, j;
    int iAalloc;
    quad q;
    pquad_t* qb_pquads;

    // ensure the arrays are large enough...
    if (qb->Nstars > qb->Ncq) {
        // (free and malloc rather than realloc because we don't care about
        //  the previous contents)
        free(qb->inbox);
        free(qb->pquads);
        qb->Ncq = qb->Nstars;
        qb->inbox =  calloc(qb->Nstars, sizeof(int));
        qb->pquads = calloc((size_t)qb->Nstars * (size_t)qb->Nstars, sizeof(pquad_t));
        if (!qb->inbox || !qb->pquads) {
            ERROR("quad-builder: failed to malloc qb->inbox or qb->pquads.  Nstars=%i.\n", qb->Nstars);
            return -1;
        }
    }

    qb_pquads = qb->pquads;

    /*
     Each time through the "for" loop below, we consider a new
     star ("newpoint").  First, we try building all quads that
     have the new star on the diagonal (star B).  Then, we try
     building all quads that have the star not on the diagonal
     (star D).

     Note that we keep the invariants iA < iB and iC < iD.
     */
    memset(&q, 0, sizeof(quad));
    for (newpoint=0; newpoint<qb->Nstars; newpoint++) {
        pquad_t* pq;
        logverb("Adding new star %i\n", newpoint);
        // quads with the new star on the diagonal:
        iB = newpoint;
        for (iA = 0; iA < newpoint; iA++) {
            pq = qb_pquads + iA*qb->Nstars + iB;
            pq->inbox = NULL;
            pq->ninbox = 0;
            pq->iA = iA;
            pq->iB = iB;

            check_scale(qb, pq);
            if (!pq->scale_ok) {
                logverb("Dropping pair %i, %i based on scale\n", newpoint, iA);
                continue;
            }

            q.star[0] = pq->staridA;
            q.star[1] = pq->staridB;

            pq->check_ok = TRUE;
            if (qb->check_AB_stars)
                pq->check_ok = qb->check_AB_stars(qb, pq, qb->check_AB_stars_token);
            if (!pq->check_ok) {
                logverb("Failed check for AB stars\n");
                continue;
            }

            // list the possible internal stars...
            ninbox = 0;
            for (iC = 0; iC < newpoint; iC++) {
                if ((iC == iA) || (iC == iB))
                    continue;
                qb->inbox[ninbox] = iC;
                ninbox++;
            }

            logverb("Number of possible internal stars for pair %i, %i: %i\n", newpoint, iA, ninbox);

            // check which ones are inside the box...
            ninbox = check_inbox(pq, qb->inbox, ninbox, qb->starxyz);
            logverb("Number of stars in the box: %i\n", ninbox);

            //if (!ninbox)
            //continue;
            if (ninbox && qb->check_internal_stars)
                ninbox = qb->check_internal_stars(qb, q.star[0], q.star[1], qb->inbox, ninbox, qb->check_internal_stars_token);
            //if (!ninbox)
            //continue;

            logverb("Number of stars in the box after checking: %i\n", ninbox);

            add_interior_stars(qb, ninbox, qb->inbox, &q, 2, qb->dimquads, 0);
            if (qb->stop_creating)
                goto theend;

            pq->inbox = malloc(qb->Nstars * sizeof(int));
            if (!pq->inbox) {
                ERROR("hpquads: failed to malloc pq->inbox.\n");
                exit(-1);
            }
            pq->ninbox = ninbox;
            memcpy(pq->inbox, qb->inbox, ninbox * sizeof(int));
            debug("iA=%i, iB=%i: saved %i 'inbox' entries.\n", iA, iB, ninbox);
        }
        iAalloc = iA;

        // quads with the new star not on the diagonal:
        iD = newpoint;
        for (iA = 0; iA < newpoint; iA++) {
            for (iB = iA + 1; iB < newpoint; iB++) {
                pq = qb_pquads + iA*qb->Nstars + iB;
                if (!(pq->scale_ok && pq->check_ok))
                    continue;
                // check if this new star is in the box.
                qb->inbox[0] = iD;
                ninbox = check_inbox(pq, qb->inbox, 1, qb->starxyz);
                if (!ninbox)
                    continue;
                if (qb->check_internal_stars)
                    ninbox = qb->check_internal_stars(qb, q.star[0], q.star[1], qb->inbox, ninbox, qb->check_internal_stars_token);
                if (!ninbox)
                    continue;

                pq->inbox[pq->ninbox] = iD;
                pq->ninbox++;

                q.star[0] = pq->staridA;
                q.star[1] = pq->staridB;

                add_interior_stars(qb, pq->ninbox, pq->inbox, &q, 2, qb->dimquads, 0);
                if (qb->stop_creating) {
                    iA = iAalloc;
                    goto theend;
                }
            }
        }
    }
 theend:
    for (i=0; i<imin(qb->Nstars, newpoint+1); i++) {
        int lim = (i == newpoint) ? iA : i;
        for (j=0; j<lim; j++) {
            pquad_t* pq = qb_pquads + j*qb->Nstars + i;
            free(pq->inbox);
            pq->inbox = NULL;
        }
    }
    return 0;
}

