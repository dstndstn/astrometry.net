/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "fitsioutils.h"
#include "anqfits.h"
#include "starkd.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"
#include "quad-builder.h"
#include "allquads.h"

allquads_t* allquads_init() {
    allquads_t* aq = calloc(1, sizeof(allquads_t));
    aq->dimquads = 4;
    return aq;
}

void allquads_free(allquads_t* aq) {
    free(aq);
}

static void add_quad(quadbuilder_t* qb, unsigned int* quad, void* token) {
    allquads_t* aq = token;
    if (log_get_level() > LOG_VERB) {
        int k;
        debug("quad: ");
        for (k=0; k<qb->dimquads; k++)
            debug("%-6i ", quad[k]);
        logverb("\n");
    }
    quad_write_const(aq->codes, aq->quads, quad, aq->starkd, qb->dimquads, aq->dimcodes);
}

static anbool check_AB(quadbuilder_t* qb, pquad_t* pq, void* token) {
    allquads_t* aq = token;
    debug("check_AB: iA=%i, iB=%i, aq->starA=%i\n", pq->iA, pq->iB, aq->starA);
    return (pq->iA == aq->starA);
}

/*
 #include "codefile.h"
 static bool check_full_quad(quadbuilder_t* qb, unsigned int* quad, int nstars, void* token) {
 allquads_t* aq = token;
 double code[4];
 double xyz[12];
 double R1, R2;
 startree_get(aq->starkd, quad[0], xyz+0);
 startree_get(aq->starkd, quad[1], xyz+3);
 startree_get(aq->starkd, quad[2], xyz+6);
 startree_get(aq->starkd, quad[3], xyz+9);
 codefile_compute_star_code(xyz, code, 4);
 logmsg("Code: %g, %g, %g, %g\n", code[0], code[1], code[2], code[3]);
 R1 = hypot(code[0] - 0.5, code[1] - 0.5);
 R2 = hypot(code[2] - 0.5, code[3] - 0.5);
 logmsg("Radii %g, %g\n", R1, R2);
 return TRUE;
 }
 */

int allquads_create_quads(allquads_t* aq) {
    quadbuilder_t* qb;
    int i, N;
    double* xyz;

    qb = quadbuilder_init();

    qb->quadd2_high = aq->quad_d2_upper;
    qb->quadd2_low  = aq->quad_d2_lower;
    qb->check_scale_high = aq->use_d2_upper;
    qb->check_scale_low  = aq->use_d2_lower;

    qb->dimquads = aq->dimquads;
    qb->add_quad = add_quad;
    qb->add_quad_token = aq;

    N = startree_N(aq->starkd);

    logverb("check scale high: %i\n", qb->check_scale_high);

    if (!qb->check_scale_high) {
        int* inds;
        inds = malloc(N * sizeof(int));
        for (i=0; i<N; i++)
            inds[i] = i;
        xyz = malloc(3 * N * sizeof(double));
        kdtree_copy_data_double(aq->starkd->tree, 0, N, xyz);

        qb->starxyz = xyz;
        qb->starinds = inds;
        qb->Nstars = N;

        quadbuilder_create(qb);

        free(xyz);
        free(inds);
    } else {
        int nq;
        int lastgrass = 0;

        /*
         xyz = malloc(3 * N * sizeof(double));
         kdtree_copy_data_double(aq->starkd->tree, 0, N, xyz);
         */

        // star A = i
        nq = aq->quads->numquads;
        for (i=0; i<N; i++) {
            double xyzA[3];
            int* inds;
            int NR;

            int grass = (i*80 / N);
            if (grass != lastgrass) {
                printf(".");
                fflush(stdout);
                lastgrass = grass;
            }

            startree_get(aq->starkd, i, xyzA);

            startree_search_for(aq->starkd, xyzA, aq->quad_d2_upper,
                                &xyz, NULL, &inds, &NR);

            /*
             startree_search_for(aq->starkd, xyzA, aq->quad_d2_upper,
             NULL, NULL, &inds, &NR);
             */

            logverb("Star %i of %i: found %i stars in range\n", i+1, N, NR);
            aq->starA = i;
            qb->starxyz = xyz;
            qb->starinds = inds;
            qb->Nstars = NR;
            qb->check_AB_stars = check_AB;
            qb->check_AB_stars_token = aq;
            //qb->check_full_quad = check_full_quad;
            //qb->check_full_quad_token = aq;

            quadbuilder_create(qb);

            logverb("Star %i of %i: wrote %i quads for this star, total %i so far.\n", i+1, N, aq->quads->numquads - nq, aq->quads->numquads);
            free(inds);
            free(xyz);
        }
        //
        //free(xyz);

        printf("\n");
    }

    quadbuilder_free(qb);
    return 0;
}

int allquads_close(allquads_t* aq) {
    startree_close(aq->starkd);

    // fix output file headers.
    if (quadfile_fix_header(aq->quads) ||
        quadfile_close(aq->quads)) {
        ERROR("Couldn't write quad output file");
        return -1;
    }
    if (codefile_fix_header(aq->codes) ||
        codefile_close(aq->codes)) {
        ERROR("Couldn't write code output file");
        return -1;
    }
    return 0;
}

int allquads_open_outputs(allquads_t* aq) {
    int hp, hpnside;
    qfits_header* hdr;

    printf("Reading star kdtree %s ...\n", aq->skdtfn);
    aq->starkd = startree_open(aq->skdtfn);
    if (!aq->starkd) {
        ERROR("Failed to open star kdtree %s\n", aq->skdtfn);
        return -1;
    }
    printf("Star tree contains %i objects.\n", startree_N(aq->starkd));

    printf("Will write to quad file %s and code file %s\n", aq->quadfn, aq->codefn);
    aq->quads = quadfile_open_for_writing(aq->quadfn);
    if (!aq->quads) {
        ERROR("Couldn't open file %s to write quads.\n", aq->quadfn);
        return -1;
    }
    aq->codes = codefile_open_for_writing(aq->codefn);
    if (!aq->codes) {
        ERROR("Couldn't open file %s to write codes.\n", aq->quadfn);
        return -1;
    }

    aq->quads->dimquads = aq->dimquads;
    aq->codes->dimcodes = aq->dimcodes;

    if (aq->id) {
        aq->quads->indexid = aq->id;
        aq->codes->indexid = aq->id;
    }

    // get the "HEALPIX" header from the skdt and put it in the code and quad headers.
    hp = qfits_header_getint(startree_header(aq->starkd), "HEALPIX", -1);
    if (hp == -1) {
        logmsg("Warning: skdt does not contain \"HEALPIX\" header.  Code and quad files will not contain this header either.\n");
    }
    aq->quads->healpix = hp;
    aq->codes->healpix = hp;
    // likewise "HPNSIDE"
    hpnside = qfits_header_getint(startree_header(aq->starkd), "HPNSIDE", 1);
    aq->quads->hpnside = hpnside;
    aq->codes->hpnside = hpnside;

    hdr = quadfile_get_header(aq->quads);
    qfits_header_add(hdr, "CXDX", "T", "All codes have the property cx<=dx.", NULL);
    qfits_header_add(hdr, "CXDXLT1", "T", "All codes have the property cx+dx<=1.", NULL);
    qfits_header_add(hdr, "MIDHALF", "T", "All codes have the property cx+dx<=1.", NULL);
    qfits_header_add(hdr, "CIRCLE", "T", "Codes live in the circle, not the box.", NULL);

    hdr = codefile_get_header(aq->codes);
    qfits_header_add(hdr, "CXDX", "T", "All codes have the property cx<=dx.", NULL);
    qfits_header_add(hdr, "CXDXLT1", "T", "All codes have the property cx+dx<=1.", NULL);
    qfits_header_add(hdr, "MIDHALF", "T", "All codes have the property cx+dx<=1.", NULL);
    qfits_header_add(hdr, "CIRCLE", "T", "Codes live in the circle, not the box.", NULL);

    if (quadfile_write_header(aq->quads)) {
        ERROR("Couldn't write headers to quads file %s\n", aq->quadfn);
        return -1;
    }
    if (codefile_write_header(aq->codes)) {
        ERROR("Couldn't write headers to code file %s\n", aq->codefn);
        return -1;
    }

    if (!aq->use_d2_lower)
        aq->quad_d2_lower = 0.0;
    if (!aq->use_d2_upper)
        aq->quad_d2_upper = 10.0;

    aq->codes->numstars = startree_N(aq->starkd);
    aq->codes->index_scale_upper = distsq2rad(aq->quad_d2_upper);
    aq->codes->index_scale_lower = distsq2rad(aq->quad_d2_lower);

    aq->quads->numstars = aq->codes->numstars;
    aq->quads->index_scale_upper = aq->codes->index_scale_upper;
    aq->quads->index_scale_lower = aq->codes->index_scale_lower;
    return 0;
}

