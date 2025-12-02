/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdint.h>
#include <assert.h>

#include "os-features.h"
#include "uniformize-catalog.h"
#include "intmap.h"
#include "healpix.h"
#include "healpix-utils.h"
#include "permutedsort.h"
#include "starutil.h"
#include "an-bool.h"
#include "mathutil.h"
#include "errors.h"
#include "log.h"
#include "boilerplate.h"
#include "fitsioutils.h"

// use 64-bit healpixes
typedef int64_t hpint;
// blocklist types
typedef ll hpl;
#define hpl_new ll_new
#define hpl_free ll_free
#define hpl_append ll_append
#define hpl_size ll_size
#define hpl_sort ll_sort

struct oh_token {
    int hp;
    hpint nside;
    hpint finenside;
};

// Return 1 if the given "hp" is outside the healpix described in "oh_token".
static int outside_healpix(hpint hp, void* vtoken) {
    struct oh_token* token = vtoken;
    hpint bighp;
    healpix_convert_nsidel(hp, token->finenside, token->nside, &bighp);
    return (bighp == token->hp ? 0 : 1);
}

static anbool is_duplicate(int hp, double ra, double dec, int Nside,
                           longmap_t* starlists,
                           double* ras, double* decs, double dedupr2) {
    double xyz[3];
    int neigh[9];
    int nn;
    double xyz2[3];
    int k;
    size_t j;
    radecdeg2xyzarr(ra, dec, xyz);
    // Check this healpix...
    neigh[0] = hp;
    // Check neighbouring healpixes... (+1 is to skip over this hp)
    nn = 1 + healpix_get_neighbours(hp, neigh+1, Nside);
    for (k=0; k<nn; k++) {
        int otherhp = neigh[k];
        bl* lst = longmap_find(starlists, otherhp, FALSE);
        if (!lst)
            continue;
        for (j=0; j<bl_size(lst); j++) {
            int otherindex;
            bl_get(lst, j, &otherindex);
            radecdeg2xyzarr(ras[otherindex], decs[otherindex], xyz2);
            if (!distsq_exceeds(xyz, xyz2, 3, dedupr2))
                return TRUE;
        }
    }
    return FALSE;
}

int uniformize_catalog(fitstable_t* intable, fitstable_t* outtable,
                       const char* racol, const char* deccol,
                       const char* sortcol, anbool sort_ascending,
                       double sort_min_cut,
                       // ?  Or do this cut in a separate process?
                       int bighp, int bignside,
                       int nmargin,
                       // uniformization nside.
                       int Nside_int,
                       double dedup_radius,
                       int nsweeps,
                       char** args, int argc) {
    anbool allsky;
    longmap_t* starlists;
    anbool dense = FALSE;
    double dedupr2 = 0.0;
    tfits_type dubl;
    int N;
    int* inorder = NULL;
    int* outorder = NULL;
    int outi;
    double *ra = NULL, *dec = NULL;
    hpl* myhps = NULL;
    int i,j,k;
    int nkeep = nsweeps;
    int noob = 0;
    int ndup = 0;
    struct oh_token token;
    int* npersweep = NULL;
    qfits_header* outhdr = NULL;
    double *sortval = NULL;

    hpint NHP;
    // up-convert Nside; which Nside will always fit in int32, we do a lot of math
    // on it where we want the results to be type hpint.
    hpint Nside = Nside_int;
    
    if (bignside == 0)
        bignside = 1;
    allsky = (bighp == -1);

    if (Nside % bignside) {
        ERROR("Fine healpixelization Nside must be a multiple of the coarse healpixelization Nside");
        return -1;
    }
    /*
     if (Nside > HP_MAX_INT_NSIDE) {
     ERROR("Error: maximum healpix Nside = %i", HP_MAX_INT_NSIDE);
     return -1;
     }
     */
    NHP = 12 * Nside * Nside;
    logverb("Healpix Nside: %lli, # healpixes on the whole sky: %lli\n", (long long)Nside, (long long)NHP);
    if (!allsky) {
        logverb("Creating index for healpix %i, nside %i\n", bighp, bignside);
        logverb("Number of healpixes: %lli\n", (long long)((Nside/bignside)*(Nside/bignside)));
    }
    logverb("Healpix side length: %g arcmin.\n", healpix_side_length_arcmin(Nside));

    dubl = fitscolumn_double_type();
    if (!racol)
        racol = "RA";
    ra = fitstable_read_column(intable, racol, dubl);
    if (!ra) {
        ERROR("Failed to find RA column (%s) in table", racol);
        return -1;
    }
    if (!deccol)
        deccol = "DEC";
    dec = fitstable_read_column(intable, deccol, dubl);
    if (!dec) {
        ERROR("Failed to find DEC column (%s) in table", deccol);
        free(ra);
        return -1;
    }

    N = fitstable_nrows(intable);
    logverb("Have %i objects\n", N);

    // FIXME -- argsort and seek around the input table, and append to
    // starlists in order; OR read from the input table in sequence and
    // sort in the starlists?
    if (sortcol) {
        logverb("Sorting by %s...\n", sortcol);
        sortval = fitstable_read_column(intable, sortcol, dubl);
        if (!sortval) {
            ERROR("Failed to read sorting column \"%s\"", sortcol);
            free(ra);
            free(dec);
            return -1;
        }
        inorder = permuted_sort(sortval, sizeof(double),
                                sort_ascending ? compare_doubles_asc : compare_doubles_desc,
                                NULL, N);
        if (sort_min_cut > -LARGE_VAL) {
            logverb("Cutting to %s > %g...\n", sortcol, sort_min_cut);
            // Cut objects with sortval < sort_min_cut.
            if (sort_ascending) {
                // skipped objects are at the front -- find the first obj
                // to keep
                for (i=0; i<N; i++)
                    if (sortval[inorder[i]] > sort_min_cut)
                        break;
                // move the "inorder" indices down.
                if (i)
                    memmove(inorder, inorder+i, (N-i)*sizeof(int));
                N -= i;
            } else {
                // skipped objects are at the end -- find the last obj to keep.
                for (i=N-1; i>=0; i--)
                    if (sortval[inorder[i]] > sort_min_cut)
                        break;
                N = i+1;
            }
            logverb("Cut to %i objects\n", N);
        }
    }

    token.nside = bignside;
    token.finenside = Nside;
    token.hp = bighp;

    if (!allsky && nmargin) {
        int bigbighp, bighpx, bighpy;
        hpl* seeds = hpl_new(256);
        logverb("Finding healpixes in range...\n");
        healpix_decompose_xy(bighp, &bigbighp, &bighpx, &bighpy, bignside);
        // Prime the queue with the fine healpixes that are on the
        // boundary of the big healpix.
        for (i=0; i<((Nside / bignside) - 1); i++) {
            // add (i,0), (i,max), (0,i), and (0,max) healpixes
            int xx = i + bighpx * (Nside / bignside);
            int yy = i + bighpy * (Nside / bignside);
            int y0 =     bighpy * (Nside / bignside);
            // -1 prevents us from double-adding the corners.
            int y1 =(1 + bighpy)* (Nside / bignside) - 1;
            int x0 =     bighpx * (Nside / bignside);
            int x1 =(1 + bighpx)* (Nside / bignside) - 1;
            assert(xx < Nside);
            assert(yy < Nside);
            assert(x0 < Nside);
            assert(x1 < Nside);
            assert(y0 < Nside);
            assert(y1 < Nside);
            hpl_append(seeds, healpix_compose_xyl(bigbighp, xx, y0, Nside));
            hpl_append(seeds, healpix_compose_xyl(bigbighp, xx, y1, Nside));
            hpl_append(seeds, healpix_compose_xyl(bigbighp, x0, yy, Nside));
            hpl_append(seeds, healpix_compose_xyl(bigbighp, x1, yy, Nside));
        }
        logmsg("Number of boundary healpixes: %zu (Nside/bignside = %lli)\n", hpl_size(seeds), (long long)Nside/bignside);

        myhps = healpix_region_searchl(-1, seeds, Nside, NULL, NULL,
                                       outside_healpix, &token, nmargin);
        logmsg("Number of margin healpixes: %zu\n", hpl_size(myhps));
        hpl_free(seeds);

        hpl_sort(myhps, TRUE);
        // DEBUG
        //il_check_consistency(myhps);
        //il_check_sorted_ascending(myhps, TRUE);
    }

    dedupr2 = arcsec2distsq(dedup_radius);
    starlists = longmap_new(sizeof(int32_t), nkeep, 0, dense);

    logverb("Placing stars in grid cells...\n");
    for (i=0; i<N; i++) {
        int hp;
        bl* lst;
        int32_t j32;
        anbool oob;
        if (inorder) {
            j = inorder[i];
            //printf("Placing star %i (%i): sort value %s = %g, RA,Dec=%g,%g\n", i, j, sortcol, sortval[j], ra[j], dec[j]);
        } else
            j = i;
		
        hp = radecdegtohealpix(ra[j], dec[j], Nside);
        //printf("HP %i\n", hp);
        // in bounds?
        oob = FALSE;
        if (myhps) {
            oob = (outside_healpix(hp, &token) && !il_sorted_contains(myhps, hp));
        } else if (!allsky) {
            oob = (outside_healpix(hp, &token));
        }
        if (oob) {
            //printf("out of bounds.\n");
            noob++;
            continue;
        }

        lst = longmap_find(starlists, hp, TRUE);
        /*
         printf("list has %i existing entries.\n", bl_size(lst));
         for (k=0; k<bl_size(lst); k++) {
         bl_get(lst, k, &j32);
         printf("  %i: index %i, %s = %g\n", k, j32, sortcol, sortval[j32]);
         }
         */

        // is this list full?
        if (nkeep && (bl_size(lst) >= nkeep)) {
            // Here we assume we're working in sorted order: once the list is full we're done.
            //printf("Skipping: list is full.\n");
            continue;
        }

        if ((dedupr2 > 0.0) &&
            is_duplicate(hp, ra[j], dec[j], Nside, starlists, ra, dec, dedupr2)) {
            //printf("Skipping: duplicate\n");
            ndup++;
            continue;
        }

        // Add the new star (by index)
        j32 = j;
        bl_append(lst, &j32);
    }
    logverb("%i outside the healpix\n", noob);
    logverb("%i duplicates\n", ndup);

    il_free(myhps);
    myhps = NULL;
    free(inorder);
    inorder = NULL;
    free(ra);
    ra = NULL;
    free(dec);
    dec = NULL;

    outorder = malloc(N * sizeof(int));
    outi = 0;

    npersweep = calloc(nsweeps, sizeof(int));

    for (k=0; k<nsweeps; k++) {
        int starti = outi;
        int32_t j32;
        for (i=0;; i++) {
            bl* lst;
            int64_t hp;
            if (!longmap_get_entry(starlists, i, &hp, &lst))
                break;
            if (bl_size(lst) <= k)
                continue;
            bl_get(lst, k, &j32);
            outorder[outi] = j32;
            //printf("sweep %i, cell #%i, hp %i, star %i, %s = %g\n", k, i, hp, j32, sortcol, sortval[j32]);
            outi++;
        }
        logmsg("Sweep %i: %i stars\n", k+1, outi - starti);
        npersweep[k] = outi - starti;

        if (sortcol) {
            // Re-sort within this sweep.
            permuted_sort(sortval, sizeof(double),
                          sort_ascending ? compare_doubles_asc : compare_doubles_desc,
                          outorder + starti, npersweep[k]);
            /*
             for (i=0; i<npersweep[k]; i++) {
             printf("  within sweep %i: star %i, j=%i, %s=%g\n",
             k, i, outorder[starti + i], sortcol, sortval[outorder[starti + i]]);
             }
             */
        }

    }
    longmap_free(starlists);
    starlists = NULL;

    //////
    free(sortval);
    sortval = NULL;

    logmsg("Total: %i stars\n", outi);
    N = outi;

    outhdr = fitstable_get_primary_header(outtable);
    if (allsky)
        qfits_header_add(outhdr, "ALLSKY", "T", "All-sky catalog.", NULL);
    BOILERPLATE_ADD_FITS_HEADERS(outhdr);
    qfits_header_add(outhdr, "HISTORY", "This file was generated by the command-line:", NULL, NULL);
    fits_add_args(outhdr, args, argc);
    qfits_header_add(outhdr, "HISTORY", "(end of command line)", NULL, NULL);
    fits_add_long_history(outhdr, "uniformize-catalog args:");
    fits_add_long_history(outhdr, "  RA,Dec columns: %s,%s", racol, deccol);
    fits_add_long_history(outhdr, "  sort column: %s", sortcol);
    fits_add_long_history(outhdr, "  sort direction: %s", sort_ascending ? "ascending" : "descending");
    if (sort_ascending)
        fits_add_long_history(outhdr, "    (ie, for mag-like sort columns)");
    else
        fits_add_long_history(outhdr, "    (ie, for flux-like sort columns)");
    fits_add_long_history(outhdr, "  uniformization nside: %i", (int)Nside);
    fits_add_long_history(outhdr, "    (ie, side length ~ %g arcmin)", healpix_side_length_arcmin(Nside));
    fits_add_long_history(outhdr, "  deduplication scale: %g arcsec", dedup_radius);
    fits_add_long_history(outhdr, "  number of sweeps: %i", nsweeps);

    fits_header_add_int(outhdr, "NSTARS", N, "Number of stars.");
    fits_header_add_int(outhdr, "HEALPIX", bighp, "Healpix covered by this catalog, with Nside=HPNSIDE");
    fits_header_add_int(outhdr, "HPNSIDE", bignside, "Nside of HEALPIX.");
    fits_header_add_int(outhdr, "CUTNSIDE", Nside, "uniformization scale (healpix nside)");
    fits_header_add_int(outhdr, "CUTMARG", nmargin, "margin size, in healpixels");
    //qfits_header_add(outhdr, "CUTBAND", cutband, "band on which the cut was made", NULL);
    fits_header_add_double(outhdr, "CUTDEDUP", dedup_radius, "deduplication radius [arcsec]");
    fits_header_add_int(outhdr, "CUTNSWEP", nsweeps, "number of sweeps");
    //fits_header_add_double(outhdr, "CUTMINMG", minmag, "minimum magnitude");
    //fits_header_add_double(outhdr, "CUTMAXMG", maxmag, "maximum magnitude");
    for (k=0; k<nsweeps; k++) {
        char key[64];
        sprintf(key, "SWEEP%i", (k+1));
        fits_header_add_int(outhdr, key, npersweep[k], "# stars added");
    }
    free(npersweep);

    if (fitstable_write_primary_header(outtable)) {
        ERROR("Failed to write primary header");
        return -1;
    }

    // Write output.
    fitstable_add_fits_columns_as_struct2(intable, outtable);
    if (fitstable_write_header(outtable)) {
        ERROR("Failed to write output table header");
        return -1;
    }
    logmsg("Writing output...\n");
    logverb("Row size: %i\n", fitstable_row_size(intable));
    if (fitstable_copy_rows_data(intable, outorder, N, outtable)) {
        ERROR("Failed to copy rows from input table to output");
        return -1;
    }
    if (fitstable_fix_header(outtable)) {
        ERROR("Failed to fix output table header");
        return -1;
    }
    free(outorder);
    return 0;
}

