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

#include "healpix.h"
#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "tic.h"
#include "fitsioutils.h"
#include "anqfits.h"
#include "permutedsort.h"
#include "bt.h"
#include "starkd.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"
#include "quad-utils.h"
#include "quad-builder.h"

// healpix type
/*
typedef il hpl;
typedef int hpint;
#define hpl_new il_new
#define hpl_insert_unique_ascending il_insert_unique_ascending
#define hpl_size il_size
#define hpl_get il_get
#define hpl_append il_append
#define hpl_free il_free
 */

typedef ll hpl;
typedef int64_t hpint;
#define hpl_new ll_new
#define hpl_insert_unique_ascending ll_insert_unique_ascending
#define hpl_size ll_size
#define hpl_get ll_get
#define hpl_append ll_append
#define hpl_free ll_free

struct hpquads {
    int dimquads;
    hpint Nside;
    startree_t* starkd;

    // bounds of quad scale, in distance between AB on the sphere.
    double quad_dist2_upper;
    double quad_dist2_lower;
    // for hp searching
    double radius2;

    bl* quadlist;
    bt* bigquadlist;

    unsigned char* nuses;

    // from find_stars():
    kdtree_qres_t* res;
    int* inds;
    double* stars;
    int Nstars;

    void* sort_data;
    int (*sort_func)(const void*, const void*);
    int sort_size;

    // for create_quad():
    anbool quad_created;
    anbool count_uses;
    hpint hp;

    // for build_quads():
    hpl* retryhps;
};
typedef struct hpquads hpquads_t;


static int compare_quads(const void* v1, const void* v2, void* token) {
    const unsigned int* q1 = v1;
    const unsigned int* q2 = v2;
    int dimquads = *(int*)token;
    int i;
    for (i=0; i<dimquads; i++) {
        if (q1[i] > q2[i])
            return 1;
        if (q1[i] < q2[i])
            return -1;
    }
    return 0;
}

static anbool find_stars(hpquads_t* me, double radius2, int R) {
    int d, j, N;
    int destind;
    double centre[3];
    int* perm;

    healpixl_to_xyzarr(me->hp, me->Nside, 0.5, 0.5, centre);
    {
        double ra,dec;
        xyzarr2radecdeg(centre, &ra, &dec);
        logverb("Find_stars: healpix center (%.5f, %.5f)\n", ra, dec);
    }
    me->res = kdtree_rangesearch_options_reuse(me->starkd->tree, me->res,
                                               centre, radius2, KD_OPTIONS_RETURN_POINTS);

    // here we could check whether stars are in the box defined by the
    // healpix boundaries plus quad scale, rather than just the circle
    // containing that box.

    N = me->res->nres;
    me->Nstars = N;
    logverb("Found %i stars near healpix center\n", N);
    if (N < me->dimquads)
        return FALSE;

    // FIXME -- could merge this step with the sorting step...

    // remove stars that have been used up.
    if (R) {
        destind = 0;
        for (j=0; j<N; j++) {
            if (me->nuses[me->res->inds[j]] >= R)
                continue;
            me->res->inds[destind] = me->res->inds[j];
            for (d=0; d<3; d++)
                me->res->results.d[destind*3+d] = me->res->results.d[j*3+d];
            destind++;
        }
        N = destind;
        if (N < me->dimquads)
            return FALSE;
    }

    // sort the stars in increasing order of index - assume
    // that this corresponds to decreasing order of brightness.

    // UNLESS another sorting is provided!

    if (me->sort_data && me->sort_func && me->sort_size) {
        /*
         Two levels of indirection here!

         me->res->inds are indices into the "sort_data" array (since kdtree is assumed to be un-permuted)

         We want to produce "perm", which permutes me->res->inds to make sort_data sorted;
         need to do this because we also want to permute results.d.

         Alternatively, we could re-fetch the results.d ...
         */
        int k;
        char* tempdata = malloc((size_t)me->sort_size * (size_t)N);
        for (k=0; k<N; k++)
            memcpy(tempdata + k*me->sort_size,
                   ((char*)me->sort_data) + me->sort_size * me->res->inds[k],
                   me->sort_size);
        perm = permuted_sort(tempdata, me->sort_size, me->sort_func, NULL, N);
        free(tempdata);

    } else {
        // find permutation that sorts by index...
        perm = permuted_sort(me->res->inds, sizeof(int), compare_ints_asc, NULL, N);
    }
    // apply the permutation...
    permutation_apply(perm, N, me->res->inds, me->res->inds, sizeof(int));
    permutation_apply(perm, N, me->res->results.d, me->res->results.d, 3 * sizeof(double));

    free(perm);

    me->inds = (int*)me->res->inds;
    me->stars = me->res->results.d;
    me->Nstars = N;

    return TRUE;
}


static anbool check_midpoint(quadbuilder_t* qb, pquad_t* pq, void* vtoken) {
    hpquads_t* me = vtoken;
    return (xyzarrtohealpixl(pq->midAB, me->Nside) == me->hp);
}

static anbool check_full_quad(quadbuilder_t* qb, unsigned int* quad, int nstars, void* vtoken) {
    hpquads_t* me = vtoken;
    anbool dup;
    if (!me->bigquadlist)
        return TRUE;
    dup = bt_contains2(me->bigquadlist, quad, compare_quads, &me->dimquads);
    return !dup;
}

static void add_quad(quadbuilder_t* qb, unsigned int* stars, void* vtoken) {
    int i;
    hpquads_t* me = vtoken;
    bl_append(me->quadlist, stars);
    if (me->count_uses) {
        for (i=0; i<me->dimquads; i++)
            me->nuses[stars[i]]++;
    }
    qb->stop_creating = TRUE;
    me->quad_created = TRUE;
}

static anbool create_quad(hpquads_t* me, anbool count_uses) {
    quadbuilder_t* qb;

    qb = quadbuilder_init();

    qb->starxyz = me->stars;
    qb->starinds = me->inds;
    qb->Nstars = me->Nstars;
    qb->dimquads = me->dimquads;
    qb->quadd2_low = me->quad_dist2_lower;
    qb->quadd2_high = me->quad_dist2_upper;
    qb->check_scale_low = TRUE;
    qb->check_scale_high = TRUE;
    qb->check_AB_stars = check_midpoint;
    qb->check_AB_stars_token = me;
    qb->check_full_quad = check_full_quad;
    qb->check_full_quad_token = me;
    qb->add_quad = add_quad;
    qb->add_quad_token = me;
    me->quad_created = FALSE;
    me->count_uses = count_uses;
    quadbuilder_create(qb);
    quadbuilder_free(qb);

    return me->quad_created;
}


static void add_headers(qfits_header* hdr, char** argv, int argc,
                        qfits_header* startreehdr, anbool circle,
                        int npasses) {
    int i;
    BOILERPLATE_ADD_FITS_HEADERS(hdr);
    qfits_header_add(hdr, "HISTORY", "This file was created by the program \"hpquads\".", NULL, NULL);
    qfits_header_add(hdr, "HISTORY", "hpquads command line:", NULL, NULL);
    fits_add_args(hdr, argv, argc);
    qfits_header_add(hdr, "HISTORY", "(end of hpquads command line)", NULL, NULL);

    qfits_header_add(startreehdr, "HISTORY", "** History entries copied from the input file:", NULL, NULL);
    fits_copy_all_headers(startreehdr, hdr, "HISTORY");
    qfits_header_add(startreehdr, "HISTORY", "** End of history entries.", NULL, NULL);

    qfits_header_add(hdr, "CXDX", "T", "All codes have the property cx<=dx.", NULL);
    qfits_header_add(hdr, "CXDXLT1", "T", "All codes have the property cx+dx<=1.", NULL);
    qfits_header_add(hdr, "MIDHALF", "T", "All codes have the property cx+dx<=1.", NULL);

    an_fits_copy_header(startreehdr, hdr, "ALLSKY");

    qfits_header_add(hdr, "CIRCLE", (circle ? "T" : "F"), 
                     (circle ? "Stars C,D live in the circle defined by AB."
                      :        "Stars C,D live in the box defined by AB."), NULL);

    // add placeholders...
    for (i=0; i<npasses; i++) {
        char key[64];
        sprintf(key, "PASS%i", i+1);
        qfits_header_add(hdr, key, "-1", "placeholder", NULL);
    }
}

static int build_quads(hpquads_t* me, hpint Nhptotry, hpl* hptotry, int R) {
    int nthispass = 0;
    hpint lastgrass = 0;
    hpint i;

    for (i=0; i<Nhptotry; i++) {
        anbool ok;
        hpint hp;
        if ((i * 80 / Nhptotry) != lastgrass) {
            printf(".");
            fflush(stdout);
            lastgrass = i * 80 / Nhptotry;
        }
        if (hptotry)
            hp = hpl_get(hptotry, i);
        else
            hp = i;
        logverb("Trying healpix %lli\n", (long long)hp);
        me->hp = hp;
        me->quad_created = FALSE;
        ok = find_stars(me, me->radius2, R);
        if (ok)
            create_quad(me, TRUE);

        if (me->quad_created)
            nthispass++;
        else {
            if (R && me->Nstars && me->retryhps)
                // there were some stars, and we're counting how many times stars are used.
                //il_insert_unique_ascending(me->retryhps, hp);
                // we don't mind hps showing up multiple times because we want to make up for the lost
                // passes during loosening...
                hpl_append(me->retryhps, hp);
            // FIXME -- could also track which hps are worth visiting in a future pass
        }
    }
    printf("\n");
    return nthispass;
}

int hpquads(startree_t* starkd,
            codefile_t* codes,
            quadfile_t* quads,
            int Nside_int,
            double scale_min_arcmin,
            double scale_max_arcmin,
            int dimquads,
            int passes,
            int Nreuses,
            int Nloosen,
            int id,
            anbool scanoccupied,

            void* sort_data,
            int (*sort_func)(const void*, const void*),
            int sort_size,
			
            char** args, int argc) {
    hpquads_t myhpquads;
    hpquads_t* me = &myhpquads;

    int i;
    int pass;
    anbool circle = TRUE;
    double radius2;
    hpl* hptotry;
    hpint Nhptotry = 0;
    int nquads;
    double hprad;
    double quadscale;

    hpint skhp, sknside;

    qfits_header* qhdr;
    qfits_header* chdr;

    int N;
    int dimcodes;
    int quadsize;
    hpint NHP;
    // Nside would fit in int, but it's often used in hpint math, so this
    // makes life easier.
    hpint Nside = Nside_int;

    memset(me, 0, sizeof(hpquads_t));

    /*
     if (Nside > HP_MAX_INT_NSIDE) {
     ERROR("Error: maximum healpix Nside = %i", HP_MAX_INT_NSIDE);
     return -1;
     }
     */
    if (Nreuses > 255) {
        ERROR("Error, reuse (-r) must be less than 256");
        return -1;
    }

    me->Nside = Nside;
    me->dimquads = dimquads;
    NHP = 12 * Nside * Nside;
    dimcodes = dimquad2dimcode(dimquads);
    quadsize = sizeof(unsigned int) * dimquads;

    logmsg("Nside=%lli.  Nside^2=%lli.  Number of healpixes=%lli.  Healpix side length ~ %g arcmin.\n",
           (long long)me->Nside, (long long)me->Nside*me->Nside, (long long)NHP, healpix_side_length_arcmin(me->Nside));

    me->sort_data = sort_data;
    me->sort_func = sort_func;
    me->sort_size = sort_size;

    tic();
    me->starkd = starkd;
    N = startree_N(me->starkd);
    logmsg("Star tree contains %i objects.\n", N);

    // get the "HEALPIX" header from the skdt...
    skhp = qfits_header_getint(startree_header(me->starkd), "HEALPIX", -1);
    if (skhp == -1) {
        if (!qfits_header_getboolean(startree_header(me->starkd), "ALLSKY", FALSE)) {
            logmsg("Warning: skdt does not contain \"HEALPIX\" header.  Code and quad files will not contain this header either.\n");
        }
    }
    // likewise "HPNSIDE"
    sknside = qfits_header_getint(startree_header(me->starkd), "HPNSIDE", 1);

    if (sknside && Nside % sknside) {
        logerr("Error: Nside (-n) must be a multiple of the star kdtree healpixelisation: %lli\n", (long long)sknside);
        return -1;
    }

    if (!scanoccupied && (N*(skhp == -1 ? 1 : sknside*sknside*12) < NHP)) {
        logmsg("\n\n");
        logmsg("NOTE, your star kdtree is sparse (has only a fraction of the stars expected)\n");
        logmsg("  so you probably will get much faster results by setting the \"-E\" command-line\n");
        logmsg("  flag.\n");
        logmsg("\n\n");
    }

    quads->dimquads = me->dimquads;
    codes->dimcodes = dimcodes;
    quads->healpix = skhp;
    codes->healpix = skhp;
    quads->hpnside = sknside;
    codes->hpnside = sknside;
    if (id) {
        quads->indexid = id;
        codes->indexid = id;
    }

    qhdr = quadfile_get_header(quads);
    chdr = codefile_get_header(codes);

    add_headers(qhdr, args, argc, startree_header(me->starkd), circle, passes);
    add_headers(chdr, args, argc, startree_header(me->starkd), circle, passes);

    if (quadfile_write_header(quads)) {
        ERROR("Couldn't write headers to quad file");
        return -1;
    }
    if (codefile_write_header(codes)) {
        ERROR("Couldn't write headers to code file");
        return -1;
    }

    quads->numstars = codes->numstars = N;
    me->quad_dist2_upper = arcmin2distsq(scale_max_arcmin);
    me->quad_dist2_lower = arcmin2distsq(scale_min_arcmin);
    codes->index_scale_upper = quads->index_scale_upper = distsq2rad(me->quad_dist2_upper);
    codes->index_scale_lower = quads->index_scale_lower = distsq2rad(me->quad_dist2_lower);
	
    me->nuses = calloc(N, sizeof(unsigned char));

    // hprad = sqrt(2) * (healpix side length / 2.)
    hprad = arcmin2dist(healpix_side_length_arcmin(Nside)) * M_SQRT1_2;
    quadscale = 0.5 * sqrt(me->quad_dist2_upper);
    // 1.01 for a bit of safety.  we'll look at a few extra stars.
    radius2 = square(1.01 * (hprad + quadscale));
    me->radius2 = radius2;

    logmsg("Healpix radius %g arcsec, quad scale %g arcsec, total %g arcsec\n",
           distsq2arcsec(hprad*hprad),
           distsq2arcsec(quadscale*quadscale),
           distsq2arcsec(radius2));

    hptotry = hpl_new(1024);

    if (scanoccupied) {
        logmsg("Scanning %i input stars...\n", N);
        for (i=0; i<N; i++) {
            double xyz[3];
            hpint j;
            if (startree_get(me->starkd, i, xyz)) {
                ERROR("Failed to get star %i", i);
                return -1;
            }
            j = xyzarrtohealpixl(xyz, Nside);
            hpl_insert_unique_ascending(hptotry, j);
            if (log_get_level() > LOG_VERB) {
                double ra,dec;
                if (startree_get_radec(me->starkd, i, &ra, &dec)) {
                    ERROR("Failed to get RA,Dec for star %i\n", i);
                    return -1;
                }
                logdebug("star %i: RA,Dec %g,%g; xyz %g,%g,%g; hp %lli\n",
                         i, ra, dec, xyz[0], xyz[1], xyz[2], (long long)j);
            }
        }
        logmsg("Will check %zu healpixes.\n", hpl_size(hptotry));
        if (log_get_level() > LOG_VERB) {
            logdebug("Checking healpixes: [ ");
            for (i=0; i<hpl_size(hptotry); i++)
                logdebug("%lli ", (long long)hpl_get(hptotry, i));
            logdebug("]\n");
        }

    } else {
        if (skhp == -1) {
            // Try all healpixes.
            il_free(hptotry);
            hptotry = NULL;
            Nhptotry = NHP;
        } else {
            // The star kdtree may itself be healpixed
            int starhp, starx, stary;
            // In that case, the healpixes we are interested in form a rectangle
            // within a big healpix.  These are the coords (in [0, Nside)) of
            // that rectangle.
            int x0, x1, y0, y1;
            int x, y;

            healpix_decompose_xy(skhp, &starhp, &starx, &stary, sknside);
            x0 =  starx    * (Nside / sknside);
            x1 = (starx+1) * (Nside / sknside);
            y0 =  stary    * (Nside / sknside);
            y1 = (stary+1) * (Nside / sknside);

            for (y=y0; y<y1; y++) {
                for (x=x0; x<x1; x++) {
                    hpint j = healpix_compose_xyl(starhp, x, y, Nside);
                    hpl_append(hptotry, j);
                }
            }
            assert(hpl_size(hptotry) == (Nside/sknside) * (Nside/sknside));
        }
    }
    if (hptotry)
        Nhptotry = hpl_size(hptotry);

    me->quadlist = bl_new(65536, quadsize);

    if (Nloosen)
        me->retryhps = hpl_new(1024);

    for (pass=0; pass<passes; pass++) {
        char key[64];
        int nthispass;

        logmsg("Pass %i of %i.\n", pass+1, passes);
        logmsg("Trying %lli healpixes.\n", (long long)Nhptotry);

        nthispass = build_quads(me, Nhptotry, hptotry, Nreuses);

        logmsg("Made %i quads (out of %lli healpixes) this pass.\n", nthispass, (long long)Nhptotry);
        logmsg("Made %i quads so far.\n", (me->bigquadlist ? bt_size(me->bigquadlist) : 0) + (int)bl_size(me->quadlist));

        sprintf(key, "PASS%i", pass+1);
        fits_header_mod_int(chdr, key, nthispass, "quads created in this pass");
        fits_header_mod_int(qhdr, key, nthispass, "quads created in this pass");

        logmsg("Merging quads...\n");
        if (!me->bigquadlist)
            me->bigquadlist = bt_new(quadsize, 256);
        for (i=0; i<bl_size(me->quadlist); i++) {
            void* q = bl_access(me->quadlist, i);
            bt_insert2(me->bigquadlist, q, FALSE, compare_quads, &me->dimquads);
        }
        bl_remove_all(me->quadlist);
    }

    hpl_free(hptotry);
    hptotry = NULL;

    if (Nloosen) {
        int R;
        for (R=Nreuses+1; R<=Nloosen; R++) {
            hpl* trylist;
            int nthispass;

            logmsg("Loosening reuse maximum to %i...\n", R);
            logmsg("Trying %zu healpixes.\n", hpl_size(me->retryhps));
            if (!hpl_size(me->retryhps))
                break;

            trylist = me->retryhps;
            me->retryhps = hpl_new(1024);
            nthispass = build_quads(me, hpl_size(trylist), trylist, R);
            logmsg("Made %i quads (out of %zu healpixes) this pass.\n", nthispass, hpl_size(trylist));
            hpl_free(trylist);
            for (i=0; i<bl_size(me->quadlist); i++) {
                void* q = bl_access(me->quadlist, i);
                bt_insert2(me->bigquadlist, q, FALSE, compare_quads, &me->dimquads);
            }
            bl_remove_all(me->quadlist);
        }
    }
    if (me->retryhps)
        hpl_free(me->retryhps);

    kdtree_free_query(me->res);
    me->res = NULL;
    me->inds = NULL;
    me->stars = NULL;
    free(me->nuses);
    me->nuses = NULL;

    logmsg("Writing quads...\n");

    // add the quads from the big-quadlist
    nquads = bt_size(me->bigquadlist);
    for (i=0; i<nquads; i++) {
        unsigned int* q = bt_access(me->bigquadlist, i);
        quad_write(codes, quads, q, me->starkd, me->dimquads, dimcodes);
    }
    // add the quads that were made during the final round.
    for (i=0; i<bl_size(me->quadlist); i++) {
        unsigned int* q = bl_access(me->quadlist, i);
        quad_write(codes, quads, q, me->starkd, me->dimquads, dimcodes);
    }

    // fix output file headers.
    if (quadfile_fix_header(quads)) {
        ERROR("Failed to fix quadfile headers");
        return -1;
    }
    if (codefile_fix_header(codes)) {
        ERROR("Failed to fix codefile headers");
        return -1;
    }

    bl_free(me->quadlist);
    bt_free(me->bigquadlist);

    toc();
    logmsg("Done.\n");
    return 0;
}

int hpquads_files(const char* skdtfn,
                  const char* codefn,
                  const char* quadfn,
                  int Nside,
                  double scale_min_arcmin,
                  double scale_max_arcmin,
                  int dimquads,
                  int passes,
                  int Nreuses,
                  int Nloosen,
                  int id,
                  anbool scanoccupied,

                  void* sort_data,
                  int (*sort_func)(const void*, const void*),
                  int sort_size,

                  char** args, int argc) {
    quadfile_t* quads;
    codefile_t* codes;
    startree_t* starkd;
    int rtn;

    logmsg("Reading star kdtree %s ...\n", skdtfn);
    starkd = startree_open(skdtfn);
    if (!starkd) {
        ERROR("Failed to open star kdtree %s\n", skdtfn);
        return -1;
    }

    logmsg("Will write to quad file %s and code file %s\n", quadfn, codefn);
    quads = quadfile_open_for_writing(quadfn);
    if (!quads) {
        ERROR("Couldn't open file %s to write quads.\n", quadfn);
        return -1;
    }
    codes = codefile_open_for_writing(codefn);
    if (!codes) {
        ERROR("Couldn't open file %s to write codes.\n", codefn);
        return -1;
    }

    rtn = hpquads(starkd, codes, quads, Nside,
                  scale_min_arcmin, scale_max_arcmin,
                  dimquads, passes, Nreuses, Nloosen, id,
                  scanoccupied, 
                  sort_data, sort_func, sort_size,
                  args, argc);
    if (rtn)
        return rtn;

    if (quadfile_close(quads)) {
        ERROR("Couldn't write quad output file");
        return -1;
    }
    if (codefile_close(codes)) {
        ERROR("Couldn't write code output file");
        return -1;
    }
    startree_close(starkd);

    return rtn;
}
