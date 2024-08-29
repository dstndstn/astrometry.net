/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 * Solve a single field
 *
 * Inputs: .ckdt .quad .skdt
 * Output: .match .rdls .wcs, ...
 */

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <libgen.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "onefield.h"
#include "tweak.h"
#include "tweak2.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "mathutil.h"
#include "quadfile.h"
#include "solvedfile.h"
#include "starkd.h"
#include "codekd.h"
#include "boilerplate.h"
#include "fitsioutils.h"
#include "verify.h"
#include "index.h"
#include "log.h"
#include "tic.h"
#include "anqfits.h"
#include "errors.h"
#include "scamp-catalog.h"
#include "permutedsort.h"
#include "bl-sort.h"

static anbool record_match_callback(MatchObj* mo, void* userdata);
static time_t timer_callback(void* user_data);
static void add_onefield_params(onefield_t* bp, qfits_header* hdr);
static void load_and_parse_wcsfiles(onefield_t* bp);
static void solve_fields(onefield_t* bp, sip_t* verify_wcs);
static void remove_invalid_fields(il* fieldlist, int maxfield);
static anbool is_field_solved(onefield_t* bp, int fieldnum);
static int write_solutions(onefield_t* bp);
static void solved_field(onefield_t* bp, int fieldnum);
static int compare_matchobjs(const void* v1, const void* v2);
static void remove_duplicate_solutions(onefield_t* bp);

// A tag-along column for index rdls / correspondence file.
struct tagalong {
    tfits_type type;
    int arraysize;
    char* name;
    char* units;
    void* data;
    // size in bytes of one item.
    int itemsize;
    int Ndata;
    // assigned by rdlist_add_tagalong_column
    int colnum;
};
typedef struct tagalong tagalong_t;

static anbool grab_tagalong_data(startree_t* starkd, MatchObj* mo, onefield_t* bp,
                                 const int* starinds, int N) {
    fitstable_t* tagalong;
    int i;
    tagalong = startree_get_tagalong(starkd);
    if (!tagalong) {
        ERROR("Failed to find tag-along table in index");
        return FALSE;
    }
    if (!mo->tagalong)
        mo->tagalong = bl_new(16, sizeof(tagalong_t));

    if (bp->rdls_tagalong_all) { // && ! bp->done_rdls_tagalong_all
        char* cols;
        // retrieve all column names.
        bp->rdls_tagalong = fitstable_get_fits_column_names(tagalong, bp->rdls_tagalong);
        cols = sl_join(bp->rdls_tagalong, ", ");
        logverb("Found tag-along columns: %s\n", cols);
        free(cols);
        //
        sl_remove_duplicates(bp->rdls_tagalong);
        cols = sl_join(bp->rdls_tagalong, ", ");
        logverb("After removing duplicates: %s\n", cols);
        free(cols);
    }
    for (i=0; i<sl_size(bp->rdls_tagalong); i++) {
        const char* col = sl_get(bp->rdls_tagalong, i);
        tagalong_t tag;
        if (fitstable_find_fits_column(tagalong, col, &(tag.units), &(tag.type), &(tag.arraysize))) {
            ERROR("Failed to find column \"%s\" in index", col);
            continue;
        }
        tag.data = fitstable_read_column_array_inds(tagalong, col, tag.type, starinds, N, NULL);
        if (!tag.data) {
            ERROR("Failed to read data for column \"%s\" in index", col);
            continue;
        }
        if ((strcaseeq(col, "ra") || strcaseeq(col, "dec")))
            asprintf_safe(&(tag.name), "%s_ref", col);
        else
            tag.name = strdup(col);
        tag.units = strdup(tag.units);
        tag.itemsize = fits_get_atom_size(tag.type) * tag.arraysize;
        tag.Ndata = N;
        bl_append(mo->tagalong, &tag);
    }
    return TRUE;
}

static anbool grab_field_tagalong_data(MatchObj* mo, xylist_t* xy, int N) {
    fitstable_t* tagalong;
    int i;
    sl* lst;
    if (!mo->field_tagalong)
        mo->field_tagalong = bl_new(16, sizeof(tagalong_t));
    tagalong = xy->table;
    lst = xylist_get_tagalong_column_names(xy, NULL);
    {
        char* txt = sl_join(lst, " ");
        logverb("Found tag-along columns from field: %s\n", txt);
        free(txt);
    }
    for (i=0; i<sl_size(lst); i++) {
        const char* col = sl_get(lst, i);
        tagalong_t tag;
        if (fitstable_find_fits_column(tagalong, col, &(tag.units), &(tag.type), &(tag.arraysize))) {
            ERROR("Failed to find column \"%s\" in index", col);
            continue;
        }
        tag.data = fitstable_read_column_array(tagalong, col, tag.type);
        if (!tag.data) {
            ERROR("Failed to read data for column \"%s\" in index", col);
            continue;
        }
        tag.name = strdup(col);
        tag.units = strdup(tag.units);
        tag.itemsize = fits_get_atom_size(tag.type) * tag.arraysize;
        tag.Ndata = N;
        bl_append(mo->field_tagalong, &tag);
    }
    sl_free2(lst);
    return TRUE;
}


/** Index handling for in_parallel and not.

 Currently it supposedly could handle both "indexnames" and "indexes",
 but we should probably just assert that only one of these can be used.
 **/
static index_t* get_index(onefield_t* bp, size_t i) {
    if (i < sl_size(bp->indexnames)) {
        char* fn = sl_get(bp->indexnames, i);
        index_t* ind = index_load(fn, bp->index_options, NULL);
        if (!ind) {
            ERROR("Failed to load index %s", fn);
            exit( -1);
        }
        return ind;
    }
    i -= sl_size(bp->indexnames);
    return pl_get(bp->indexes, i);
}
static char* get_index_name(onefield_t* bp, size_t i) {
    index_t* index;
    if (i < sl_size(bp->indexnames)) {
        char* fn = sl_get(bp->indexnames, i);
        return fn;
    }
    i -= sl_size(bp->indexnames);
    index = pl_get(bp->indexes, i);
    return index->indexname;
}
static void done_with_index(onefield_t* bp, size_t i, index_t* ind) {
    if (i < sl_size(bp->indexnames)) {
        index_close(ind);
    }
}
static size_t n_indexes(onefield_t* bp) {
    return sl_size(bp->indexnames) + pl_size(bp->indexes);
}



void onefield_clear_verify_wcses(onefield_t* bp) {
    bl_remove_all(bp->verify_wcs_list);
}

void onefield_clear_solutions(onefield_t* bp) {
    bl_remove_all(bp->solutions);
}

void onefield_clear_indexes(onefield_t* bp) {
    sl_remove_all(bp->indexnames);
    pl_remove_all(bp->indexes);
}

void onefield_set_field_file(onefield_t* bp, const char* fn) {
    free(bp->fieldfname);
    bp->fieldfname = strdup_safe(fn);
}

void onefield_set_solved_file(onefield_t* bp, const char* fn) {
    onefield_set_solvedin_file (bp, fn);
    onefield_set_solvedout_file(bp, fn);
}

void onefield_set_solvedin_file(onefield_t* bp, const char* fn) {
    free(bp->solved_in);
    bp->solved_in = strdup_safe(fn);
}

void onefield_set_solvedout_file(onefield_t* bp, const char* fn) {
    free(bp->solved_out);
    bp->solved_out = strdup_safe(fn);
}

void onefield_set_cancel_file(onefield_t* bp, const char* fn) {
    free(bp->cancelfname);
    bp->cancelfname = strdup_safe(fn);
}

void onefield_set_match_file(onefield_t* bp, const char* fn) {
    free(bp->matchfname);
    bp->matchfname = strdup_safe(fn);
}

void onefield_set_rdls_file(onefield_t* bp, const char* fn) {
    free(bp->indexrdlsfname);
    bp->indexrdlsfname = strdup_safe(fn);
}

void onefield_set_scamp_file(onefield_t* bp, const char* fn) {
    free(bp->scamp_fname);
    bp->scamp_fname = strdup_safe(fn);
}

void onefield_set_corr_file(onefield_t* bp, const char* fn) {
    free(bp->corr_fname);
    bp->corr_fname = strdup_safe(fn);
}

void onefield_set_wcs_file(onefield_t* bp, const char* fn) {
    free(bp->wcs_template);
    bp->wcs_template = strdup_safe(fn);
}

void onefield_set_xcol(onefield_t* bp, const char* x) {
    free(bp->xcolname);
    if (!x)
        x = "X";
    bp->xcolname = strdup(x);
}

void onefield_set_ycol(onefield_t* bp, const char* y) {
    free(bp->ycolname);
    if (!y)
        y = "Y";
    bp->ycolname = strdup_safe(y);
}

void onefield_add_index(onefield_t* bp, const char* index) {
    sl_append(bp->indexnames, index);
}

void onefield_add_loaded_index(onefield_t* bp, index_t* ind) {
    pl_append(bp->indexes, ind);
}

void onefield_add_verify_wcs(onefield_t* bp, sip_t* wcs) {
    bl_append(bp->verify_wcs_list, wcs);
}

void onefield_add_field(onefield_t* bp, int field) {
    il_insert_unique_ascending(bp->fieldlist, field);
}

void onefield_add_field_range(onefield_t* bp, int lo, int hi) {
    int i;
    for (i=lo; i<=hi; i++) {
        il_insert_unique_ascending(bp->fieldlist, i);
    }
}

static void check_time_limits(onefield_t* bp) {
    if (bp->total_timelimit || bp->timelimit) {
        double now = timenow();
        if (bp->total_timelimit && (now - bp->time_total_start > bp->total_timelimit)) {
            logmsg("Total wall-clock time limit reached!\n");
            bp->hit_total_timelimit = TRUE;
        }
        if (bp->timelimit && (now - bp->time_start > bp->timelimit)) {
            logmsg("Wall-clock time limit reached!\n");
            bp->hit_timelimit = TRUE;
        }
    }
    if (bp->total_cpulimit || bp->cpulimit) {
        float now = get_cpu_usage();
        if ((bp->total_cpulimit > 0.0) &&
            (now - bp->cpu_total_start > bp->total_cpulimit)) {
            logmsg("Total CPU time limit reached!\n");
            bp->hit_total_cpulimit = TRUE;
        }
        if ((bp->cpulimit > 0.0) &&
            (now - bp->cpu_start > bp->cpulimit)) {
            logmsg("CPU time limit reached!\n");
            bp->hit_cpulimit = TRUE;
        }
    }
    if (bp->hit_total_timelimit ||
        bp->hit_total_cpulimit ||
        bp->hit_timelimit ||
        bp->hit_cpulimit)
        bp->solver.quit_now = TRUE;
}

void onefield_run(onefield_t* bp) {
    solver_t* sp = &(bp->solver);
    size_t i, I;
    size_t Nindexes;

    // Record current time for total wall-clock time limit.
    bp->time_total_start = timenow();

    // Record current CPU usage for total cpu-usage limit.
    bp->cpu_total_start = get_cpu_usage();

    // Parse WCS files submitted for verification.
    load_and_parse_wcsfiles(bp);

    // Read .xyls file...
    logverb("Reading fields file %s...", bp->fieldfname);
    bp->xyls = xylist_open(bp->fieldfname);
    if (!bp->xyls) {
        ERROR("Failed to read xylist.\n");
        exit( -1);
    }
    xylist_set_xname(bp->xyls, bp->xcolname);
    xylist_set_yname(bp->xyls, bp->ycolname);
    xylist_set_include_flux(bp->xyls, FALSE);
    xylist_set_include_background(bp->xyls, FALSE);
    logverb("found %u fields.\n", xylist_n_fields(bp->xyls));

    remove_invalid_fields(bp->fieldlist, xylist_n_fields(bp->xyls));

    Nindexes = n_indexes(bp);

    // Verify any WCS estimates we have.
    if (bl_size(bp->verify_wcs_list)) {
        int i;
        int w;

        // We want to get the best logodds out of all the indices, so we set the
        // logodds-to-solve impossibly high so that a "good enough" solution doesn't
        // stop us from continuing to search...
        double oldodds = bp->logratio_tosolve;
        bp->logratio_tosolve = LARGE_VAL;

        for (w = 0; w < bl_size(bp->verify_wcs_list); w++) {
            double pixscale;
            double quadlo, quadhi;
            sip_t* wcs = bl_access(bp->verify_wcs_list, w);

            // We don't want to try to verify a wide-field image using a narrow-
            // field index, because it will contain a TON of index stars in the
            // field.  We therefore only try to verify using indices that contain
            // quads that could have been found in the image.
            if (wcs->wcstan.imagew == 0.0 && sp->field_maxx > 0.0)
                wcs->wcstan.imagew = sp->field_maxx;
            if (wcs->wcstan.imageh == 0.0 && sp->field_maxy > 0.0)
                wcs->wcstan.imageh = sp->field_maxy;

            if ((wcs->wcstan.imagew == 0) ||
                (wcs->wcstan.imageh == 0)) {
                logmsg("Verifying WCS: image width or height is zero / unknown.\n");
                continue;
            }
            pixscale = sip_pixel_scale(wcs);
            quadlo = bp->quad_size_fraction_lo
                * MIN(wcs->wcstan.imagew, wcs->wcstan.imageh)
                * pixscale;
            quadhi = bp->quad_size_fraction_hi
                * MAX(wcs->wcstan.imagew, wcs->wcstan.imageh)
                * pixscale;
            logmsg("Verifying WCS using indices with quads of size [%g, %g] arcmin\n",
                   arcsec2arcmin(quadlo), arcsec2arcmin(quadhi));

            for (I=0; I<Nindexes; I++) {
                index_t* index = get_index(bp, I);
                if (!index_overlaps_scale_range(index, quadlo, quadhi)) {
                    done_with_index(bp, I, index);
                    continue;
                }
                solver_add_index(sp, index);
                sp->index = index;
                logmsg("Verifying WCS with index %zu of %zu (%s)\n",  I + 1, Nindexes, index->indexname);
                // Do it!
                solve_fields(bp, wcs);
                // Clean up this index...
                done_with_index(bp, I, index);
                solver_clear_indexes(sp);
            }
        }

        bp->logratio_tosolve = oldodds;

        logmsg("Got %zu solutions.\n", bl_size(bp->solutions));

        if (bp->best_hit_only)
            remove_duplicate_solutions(bp);

        for (i=0; i<bl_size(bp->solutions); i++) {
            MatchObj* mo = bl_access(bp->solutions, i);
            if (mo->logodds >= bp->logratio_tosolve)
                solved_field(bp, mo->fieldnum);
        }
    }

    if (bp->single_field_solved)
        goto cleanup;

    // Start solving...
    if (bp->indexes_inparallel) {

        // Add all the indexes...
        for (I=0; I<Nindexes; I++) {
            index_t* index = get_index(bp, I);
            solver_add_index(sp, index);
        }

        // Record current CPU usage.
        bp->cpu_start = get_cpu_usage();
        // Record current wall-clock time.
        bp->time_start = time(NULL);

        // Do it!
        solve_fields(bp, NULL);

        // Clean up the indices...
        for (I=0; I<Nindexes; I++) {
            index_t* index = get_index(bp, I);
            done_with_index(bp, I, index);
        }
        solver_clear_indexes(sp);

    } else {

        for (I=0; I<Nindexes; I++) {
            index_t* index;

            if (bp->hit_total_timelimit || bp->hit_total_cpulimit)
                break;
            if (bp->single_field_solved)
                break;
            if (bp->cancelled)
                break;

            // Load the index...
            index = get_index(bp, I);
            solver_add_index(sp, index);
            logverb("Trying index %s...\n", index->indexname);

            // Record current CPU usage.
            bp->cpu_start = get_cpu_usage();
            // Record current wall-clock time.
            bp->time_start = time(NULL);

            // Do it!
            solve_fields(bp, NULL);

            // Clean up this index...
            done_with_index(bp, I, index);
            solver_clear_indexes(sp);
        }
    }

 cleanup:
    // Clean up.
    xylist_close(bp->xyls);

    if (write_solutions(bp))
        exit(-1);

    for (i=0; i<bl_size(bp->solutions); i++) {
        MatchObj* mo = bl_access(bp->solutions, i);
        verify_free_matchobj(mo);
        onefield_free_matchobj(mo);
    }
    bl_remove_all(bp->solutions);
}

void onefield_init(onefield_t* bp) {
    // Reset params.
    memset(bp, 0, sizeof(onefield_t));

    bp->fieldlist = il_new(256);
    bp->solutions = bl_new(16, sizeof(MatchObj));
    bp->indexnames = sl_new(16);
    bp->indexes = pl_new(16);
    bp->verify_wcs_list = bl_new(1, sizeof(sip_t));
    bp->verify_wcsfiles = sl_new(1);
    bp->fieldid_key = strdup("FIELDID");
    onefield_set_xcol(bp, NULL);
    onefield_set_ycol(bp, NULL);
    bp->quad_size_fraction_lo = DEFAULT_QSF_LO;
    bp->quad_size_fraction_hi = DEFAULT_QSF_HI;
    bp->nsolves = 1;

    bp->xyls_tagalong_all = TRUE;
    // don't set sp-> here because solver_set_default_values()
    // will get called next and wipe it out...
}

int onefield_parameters_are_okay(onefield_t* bp, solver_t* sp) {
    if (sp->distractor_ratio == 0) {
        logerr("You must set a \"distractors\" proportion.\n");
        return 0;
    }
    if (!(sl_size(bp->indexnames) || (bp->indexes_inparallel && pl_size(bp->indexes)))) {
        logerr("You must specify one or more indexes.\n");
        return 0;
    }
    if (!bp->fieldfname) {
        logerr("You must specify a field filename (xylist).\n");
        return 0;
    }
    if (sp->codetol < 0.0) {
        logerr("You must specify codetol > 0\n");
        return 0;
    }
    if (sp->verify_pix <= 0.0) {
        logerr("You must specify a positive verify_pix.\n");
        return 0;
    }
    if ((sp->funits_lower != 0.0) && (sp->funits_upper != 0.0) &&
        (sp->funits_lower > sp->funits_upper)) {
        logerr("fieldunits_lower MUST be less than fieldunits_upper.\n");
        logerr("\n(in other words, the lower-bound of scale estimate must "
               "be less than the upper-bound!)\n\n");
        return 0;
    }
    return 1;
}

int onefield_is_run_obsolete(onefield_t* bp, solver_t* sp) {
    // If we're just solving one field, check to see if it's already
    // solved before doing a bunch of work and spewing tons of output.
    if ((il_size(bp->fieldlist) == 1) && bp->solved_in) {
        if (is_field_solved(bp, il_get(bp->fieldlist, 0)))
            return 1;
    }
    // Early check to see if this job was cancelled.
    if (bp->cancelfname) {
        if (file_exists(bp->cancelfname)) {
            logerr("Run cancelled.\n");
            return 1;
        }
    }

    return 0;
}

static void load_and_parse_wcsfiles(onefield_t* bp) {
    int i;
    for (i = 0; i < sl_size(bp->verify_wcsfiles); i++) {
        sip_t wcs;
        char* fn = sl_get(bp->verify_wcsfiles, i);
        logmsg("Reading WCS header to verify from file %s\n", fn);
        memset(&wcs, 0, sizeof(sip_t));
        if (!sip_read_header_file(fn, &wcs)) {
            logerr("Failed to parse WCS header from file %s\n", fn);
            continue;
        }
        bl_append(bp->verify_wcs_list, &wcs);
    }
}

void onefield_log_run_parameters(onefield_t* bp) {
    solver_t* sp = &(bp->solver);
    int i, N;

    logverb("solver run parameters:\n");
    logverb("indexes:\n");
    N = n_indexes(bp);
    for (i=0; i<N; i++)
        logverb("  %s\n", get_index_name(bp, i));
    if (bp->fieldfname)
        logverb("fieldfname %s\n", bp->fieldfname);
    logverb("fields ");
    for (i = 0; i < il_size(bp->fieldlist); i++)
        logverb("%i ", il_get(bp->fieldlist, i));
    logverb("\n");
    for (i = 0; i < sl_size(bp->verify_wcsfiles); i++)
        logverb("verify %s\n", sl_get(bp->verify_wcsfiles, i));
    logverb("fieldid %i\n", bp->fieldid);
    if (bp->matchfname)
        logverb("matchfname %s\n", bp->matchfname);
    if (bp->solved_in)
        logverb("solved_in %s\n", bp->solved_in);
    if (bp->solved_out)
        logverb("solved_out %s\n", bp->solved_out);
    if (bp->cancelfname)
        logverb("cancel %s\n", bp->cancelfname);
    if (bp->wcs_template)
        logverb("wcs %s\n", bp->wcs_template);
    if (bp->fieldid_key)
        logverb("fieldid_key %s\n", bp->fieldid_key);
    if (bp->indexrdlsfname)
        logverb("indexrdlsfname %s\n", bp->indexrdlsfname);
    logverb("parity %i\n", sp->parity);
    logverb("codetol %g\n", sp->codetol);
    logverb("startdepth %i\n", sp->startobj);
    logverb("enddepth %i\n", sp->endobj);
    logverb("fieldunits_lower %g\n", sp->funits_lower);
    logverb("fieldunits_upper %g\n", sp->funits_upper);
    logverb("verify_pix %g\n", sp->verify_pix);
    if (bp->xcolname)
        logverb("xcolname %s\n", bp->xcolname);
    if (bp->ycolname)
        logverb("ycolname %s\n", bp->ycolname);
    logverb("maxquads %i\n", sp->maxquads);
    logverb("maxmatches %i\n", sp->maxmatches);
    logverb("cpulimit %f\n", bp->cpulimit);
    logverb("timelimit %i\n", bp->timelimit);
    logverb("total_timelimit %g\n", bp->total_timelimit);
    logverb("total_cpulimit %f\n", bp->total_cpulimit);
}

void onefield_cleanup(onefield_t* bp) {
    il_free(bp->fieldlist);
    bl_free(bp->solutions);
    sl_free2(bp->indexnames);
    pl_free(bp->indexes);
    sl_free2(bp->verify_wcsfiles);
    bl_free(bp->verify_wcs_list);
    sl_free2(bp->rdls_tagalong);

    free(bp->cancelfname);
    free(bp->fieldfname);
    free(bp->fieldid_key);
    free(bp->indexrdlsfname);
    free(bp->scamp_fname);
    free(bp->corr_fname);
    free(bp->matchfname);
    free(bp->solved_in);
    free(bp->solved_out);
    free(bp->wcs_template);
    free(bp->xcolname);
    free(bp->ycolname);
    free(bp->sort_rdls);
}

static int sort_rdls(MatchObj* mymo, onefield_t* bp) {
    const solver_t* sp = &(bp->solver);
    anbool asc = TRUE;
    char* colname = bp->sort_rdls;
    double* sortdata;
    fitstable_t* tagalong;
    int* perm;
    int i;
    logverb("Sorting RDLS by column \"%s\"\n", bp->sort_rdls);
    if (colname[0] == '-') {
        colname++;
        asc = FALSE;
    }
    tagalong = startree_get_tagalong(sp->index->starkd);
    if (!tagalong) {
        ERROR("Failed to find tag-along table in index");
        return -1;
    }
    sortdata = fitstable_read_column_inds(tagalong, colname, fitscolumn_double_type(),
                                          mymo->refstarid, mymo->nindex);
    if (!sortdata) {
        ERROR("Failed to read data for column \"%s\" in index", colname);
        return -1;
    }
    perm = permutation_init(NULL, mymo->nindex);
    permuted_sort(sortdata, sizeof(double), asc ? compare_doubles_asc : compare_doubles_desc,
                  perm, mymo->nindex);
    free(sortdata);

    if (mymo->refxyz)
        permutation_apply(perm, mymo->nindex, mymo->refxyz, mymo->refxyz, 3*sizeof(double));
    // probably not set yet, but what the heck...
    if (mymo->refradec)
        permutation_apply(perm, mymo->nindex, mymo->refradec,  mymo->refradec, 2*sizeof(double));
    if (mymo->refxy)
        permutation_apply(perm, mymo->nindex, mymo->refxy,     mymo->refxy,    2*sizeof(double));
    if (mymo->refstarid)
        permutation_apply(perm, mymo->nindex, mymo->refstarid, mymo->refstarid,  sizeof(int));
    if (mymo->theta)
        for (i=0; i<mymo->nfield; i++) {
            if (mymo->theta[i] < 0)
                continue;
            mymo->theta[i] = perm[mymo->theta[i]];
        }
    free(perm);
    return 0;
}

static anbool record_match_callback(MatchObj* mo, void* userdata) {
    onefield_t* bp = userdata;
    solver_t* sp = &(bp->solver);
    MatchObj* mymo;
    int ind;

    check_time_limits(bp);

    // Copy "mo" to "mymo".
    ind = bl_insert_sorted(bp->solutions, mo, compare_matchobjs);
    mymo = bl_access(bp->solutions, ind);

    // steal these arrays from "mo" (prevent them from being free()'d
    // by the caller)
    mo->theta = NULL;
    mo->matchodds = NULL;
    mo->refxyz = NULL;
    mo->refxy = NULL;
    mo->refstarid = NULL;
    mo->testperm = NULL;

    // We have no guarantee that the index will still be open when it
    // comes time to write our output files, so we've got to grab everything
    // we need now while it's at hand.

    if (bp->indexrdlsfname || bp->scamp_fname || bp->corr_fname) {
        int i;

        // This must happen first, because it reorders the "ref" arrays,
        // and we want that to be done before more data are integrated.
        if (bp->sort_rdls) {
            if (sort_rdls(mymo, bp)) {
                ERROR("Failed to sort RDLS file by column \"%s\"", bp->sort_rdls);
            }
        }

        logdebug("Converting %i reference stars from xyz to radec\n", mymo->nindex);
        mymo->refradec = malloc(mymo->nindex * 2 * sizeof(double));
        for (i=0; i<mymo->nindex; i++) {
            xyzarr2radecdegarr(mymo->refxyz+i*3, mymo->refradec+i*2);
            logdebug("  %i: radec %.2f,%.2f\n", i, mymo->refradec[i*2], mymo->refradec[i*2+1]);
        }

        mymo->fieldxy = malloc(mymo->nfield * 2 * sizeof(double));
        // whew!
        memcpy(mymo->fieldxy, bp->solver.vf->xy, mymo->nfield * 2 * sizeof(double));

        // Tweak was here...

        // FIXME -- add MAG, MAGERR, and positional errors for SCAMP catalog.

        if (bp->rdls_tagalong || bp->rdls_tagalong_all)
            grab_tagalong_data(sp->index->starkd, mymo, bp, mymo->refstarid, mymo->nindex);

        // FIXME -- we don't support specifying individual fields (yet)
        assert(bp->xyls_tagalong_all);
        assert(!bp->xyls_tagalong);
        if (bp->xyls_tagalong_all)
            grab_field_tagalong_data(mymo, bp->xyls, mymo->nfield);
    }

    if (mymo->logodds < bp->logratio_tosolve)
        return FALSE;

    // this match is considered a solution.

    bp->nsolves_sofar++;
    if (bp->nsolves_sofar < bp->nsolves) {
        logmsg("Found a quad that solves the image; that makes %i of %i required.\n",
               bp->nsolves_sofar, bp->nsolves);
    } else {
        if (bp->solver.index) {
            char* base = basename_safe(bp->solver.index->indexname);
            logmsg("Field %i: solved with index %s.\n", mymo->fieldnum, base);
            free(base);
        } else {
            logmsg("Field %i: solved with index %i", mymo->fieldnum, mymo->indexid);
            if (mymo->healpix >= 0)
                logmsg(", healpix %i\n", mymo->healpix);
            else
                logmsg("\n");
        }
        return TRUE;
    }
    return FALSE;
}

static time_t timer_callback(void* user_data) {
    onefield_t* bp = user_data;

    check_time_limits(bp);

    // check if the field has already been solved...
    if (is_field_solved(bp, bp->fieldnum))
        return 0;
    if (bp->cancelfname && file_exists(bp->cancelfname)) {
        bp->cancelled = TRUE;
        logmsg("File \"%s\" exists: cancelling.\n", bp->cancelfname);
        return 0;
    }
    return 1; // wait 1 second... FIXME config?
}

static void add_onefield_params(onefield_t* bp, qfits_header* hdr) {
    solver_t* sp = &(bp->solver);
    int i;
    int Nindexes;
    fits_add_long_comment(hdr, "-- onefield solver parameters: --");
    if (sp->index) {
        fits_add_long_comment(hdr, "Index name: %s", sp->index->indexname?sp->index->indexname:"(null)");
        fits_add_long_comment(hdr, "Index id: %i", sp->index->indexid);
        fits_add_long_comment(hdr, "Index healpix: %i", sp->index->healpix);
        fits_add_long_comment(hdr, "Index healpix nside: %i", sp->index->hpnside);
        fits_add_long_comment(hdr, "Index scale lower: %g arcsec", sp->index->index_scale_lower);
        fits_add_long_comment(hdr, "Index scale upper: %g arcsec", sp->index->index_scale_upper);
        fits_add_long_comment(hdr, "Index jitter: %g", sp->index->index_jitter);
        fits_add_long_comment(hdr, "Circle: %s", sp->index->circle ? "yes" : "no");
        fits_add_long_comment(hdr, "Cxdx margin: %g", sp->cxdx_margin);
    }
    Nindexes = n_indexes(bp);
    for (i = 0; i < Nindexes; i++)
        fits_add_long_comment(hdr, "Index(%i): %s", i, get_index_name(bp, i)?get_index_name(bp, i):"(null)");

    fits_add_long_comment(hdr, "Field name: %s", bp->fieldfname?bp->fieldfname:"(null)");
    fits_add_long_comment(hdr, "Field scale lower: %g arcsec/pixel", sp->funits_lower);
    fits_add_long_comment(hdr, "Field scale upper: %g arcsec/pixel", sp->funits_upper);
    fits_add_long_comment(hdr, "X col name: %s", bp->xcolname?bp->xcolname:"(null)");
    fits_add_long_comment(hdr, "Y col name: %s", bp->ycolname?bp->ycolname:"(null)");
    fits_add_long_comment(hdr, "Start obj: %i", sp->startobj);
    fits_add_long_comment(hdr, "End obj: %i", sp->endobj);
	
    // 'Solved_in' is often a NULL pointer.
    // If %s is a NULL pointer, vasprintf() causes a segmentation fault (due to strlen()) on Solaris -> added treatment of this case for portability. 
    // GNU/Linux implementation of vasprintf() catches NULL pointer and prints "(null)" in header. Seems to be an issue on Solaris only.
    fits_add_long_comment(hdr, "Solved_in: %s", bp->solved_in?bp->solved_in:"(null)");
    fits_add_long_comment(hdr, "Solved_out: %s", bp->solved_out?bp->solved_out:"(null)");

    fits_add_long_comment(hdr, "Parity: %i", sp->parity);
    fits_add_long_comment(hdr, "Codetol: %g", sp->codetol);
    fits_add_long_comment(hdr, "Verify pixels: %g pix", sp->verify_pix);

    fits_add_long_comment(hdr, "Maxquads: %i", sp->maxquads);
    fits_add_long_comment(hdr, "Maxmatches: %i", sp->maxmatches);
    fits_add_long_comment(hdr, "Cpu limit: %f s", bp->cpulimit);
    fits_add_long_comment(hdr, "Time limit: %i s", bp->timelimit);
    fits_add_long_comment(hdr, "Total time limit: %g s", bp->total_timelimit);
    fits_add_long_comment(hdr, "Total CPU limit: %f s", bp->total_cpulimit);

    fits_add_long_comment(hdr, "Tweak: %s", (sp->do_tweak ? "yes" : "no"));
    if (sp->do_tweak) {
        fits_add_long_comment(hdr, "Tweak AB order: %i", sp->tweak_aborder);
        fits_add_long_comment(hdr, "Tweak ABP order: %i", sp->tweak_abporder);
    }

    fits_add_long_comment(hdr, "--");
}

static void remove_invalid_fields(il* fieldlist, int maxfield) {
    int i;
    for (i=0; i<il_size(fieldlist); i++) {
        int fieldnum = il_get(fieldlist, i);
        if (fieldnum >= 1 && fieldnum <= maxfield)
            continue;
        if (fieldnum > maxfield) {
            logerr("Field %i does not exist (max=%i).\n", fieldnum, maxfield);
        }
        if (fieldnum < 1) {
            logerr("Field %i is invalid (must be >= 1).\n", fieldnum);
        }
        il_remove(fieldlist, i);
        i--;
    }
}

static void solve_fields(onefield_t* bp, sip_t* verify_wcs) {
    solver_t* sp = &(bp->solver);
    double last_utime, last_stime;
    double utime, stime;
    struct timeval wtime, last_wtime;
    int fi;

    get_resource_stats(&last_utime, &last_stime, NULL);
    gettimeofday(&last_wtime, NULL);

    for (fi = 0; fi < il_size(bp->fieldlist); fi++) {
        int fieldnum;
        MatchObj template ;
        qfits_header* fieldhdr = NULL;

        fieldnum = il_get(bp->fieldlist, fi);

        memset(&template, 0, sizeof(MatchObj));
        template.fieldnum = fieldnum;
        template.fieldfile = bp->fieldid;

        // Get the FIELDID string from the xyls FITS header.
        if (xylist_open_field(bp->xyls, fieldnum)) {
            logerr("Failed to open extension %i in xylist.\n", fieldnum);
            goto cleanup;
        }
        fieldhdr = xylist_get_header(bp->xyls);
        if (fieldhdr) {
            char* idstr = fits_get_dupstring(fieldhdr, bp->fieldid_key);
            if (idstr)
                strncpy(template.fieldname, idstr, sizeof(template.fieldname) - 1);
            free(idstr);
        }

        // Has the field already been solved?
        if (is_field_solved(bp, fieldnum))
            goto cleanup;

        // Get the field.
        solver_set_field(sp, xylist_read_field(bp->xyls, NULL));
        if (!sp->fieldxy_orig) {
            logerr("Failed to read xylist field.\n");
            goto cleanup;
        }

        solver_reset_counters(sp);
        solver_reset_best_match(sp);

        sp->mo_template = &template;
        sp->record_match_callback = record_match_callback;
        sp->timer_callback = timer_callback;
        sp->userdata = bp;

        bp->fieldnum = fieldnum;
        bp->nsolves_sofar = 0;

        solver_preprocess_field(sp);

        if (verify_wcs) {
            //MatchObj mo;
            logmsg("Verifying WCS of field %i.\n", fieldnum);
            solver_verify_sip_wcs(sp, verify_wcs); //, &mo);
            logmsg(" --> log-odds %g\n", sp->best_logodds);

        } else {
            logverb("Solving field %i.\n", fieldnum);
            sp->distance_from_quad_bonus = TRUE;
            solver_log_params(sp);

            // The real thing
            solver_run(sp);

            logverb("Field %i: tried %i quads, matched %i codes.\n",
                    fieldnum, sp->numtries, sp->nummatches);

            if (sp->maxquads && sp->numtries >= sp->maxquads)
                logmsg("  exceeded the number of quads to try: %i >= %i.\n",
                       sp->numtries, sp->maxquads);
            if (sp->maxmatches && sp->nummatches >= sp->maxmatches)
                logmsg("  exceeded the number of quads to match: %i >= %i.\n",
                       sp->nummatches, sp->maxmatches);
            if (bp->cancelled)
                logmsg("  cancelled at user request.\n");
        }


        if (sp->best_match_solves) {
            solved_field(bp, fieldnum);
        } else if (!verify_wcs) {
            // Field unsolved.
            logerr("Field %i did not solve", fieldnum);
            if (bp->solver.index && bp->solver.index->indexname) {
                char* copy;
                char* base;
                copy = strdup(bp->solver.index->indexname);
                base = strdup(basename(copy));
                free(copy);
                logerr(" (index %s", base);
                free(base);
                if (bp->solver.endobj)
                    logerr(", field objects %i-%i", bp->solver.startobj+1, bp->solver.endobj);
                logerr(")");
            }
            logerr(".\n");
            if (sp->have_best_match) {
                logverb("Best match encountered: ");
                matchobj_print(&(sp->best_match), log_get_level());
            } else {
                logverb("Best odds encountered: %g\n", exp(sp->best_logodds));
            }
        }

        solver_free_field(sp);

        get_resource_stats(&utime, &stime, NULL);
        gettimeofday(&wtime, NULL);
        logverb("Spent %g s user, %g s system, %g s total, %g s wall time.\n",
                (utime - last_utime), (stime - last_stime),
                (stime - last_stime + utime - last_utime),
                millis_between(&last_wtime, &wtime) * 0.001);

        last_utime = utime;
        last_stime = stime;
        last_wtime = wtime;

    cleanup:
        solver_cleanup_field(sp);
    }
}

static anbool is_field_solved(onefield_t* bp, int fieldnum) {
    anbool solved = FALSE;
    if (bp->solved_in) {
        solved = solvedfile_get(bp->solved_in, fieldnum);
        logverb("Checking %s file %i to see if the field is solved: %s.\n",
                bp->solved_in, fieldnum, (solved ? "yes" : "no"));
    }
    if (solved) {
        // file exists; field has already been solved.
        logmsg("Field %i: solvedfile %s: field has been solved.\n", fieldnum, bp->solved_in);
        return TRUE;
    }
    return FALSE;
}

static void solved_field(onefield_t* bp, int fieldnum) {
    // Record in solved file, or send to solved server.
    if (bp->solved_out) {
        logmsg("Field %i solved: writing to file %s to indicate this.\n", fieldnum, bp->solved_out);
        if (solvedfile_set(bp->solved_out, fieldnum)) {
            logerr("Failed to write solvedfile %s.\n", bp->solved_out);
        }
    }
    // If we're just solving a single field, and we solved it...
    if (il_size(bp->fieldlist) == 1)
        bp->single_field_solved = TRUE;
}

void onefield_matchobj_deep_copy(const MatchObj* mo, MatchObj* dest) {
    if (!mo || !dest)
        return;
    if (mo->sip) {
        dest->sip = sip_create();
        memcpy(dest->sip, mo->sip, sizeof(sip_t));
    }
    if (mo->refradec) {
        dest->refradec = malloc(mo->nindex * 2 * sizeof(double));
        memcpy(dest->refradec, mo->refradec, mo->nindex * 2 * sizeof(double));
    }
    if (mo->fieldxy) {
        dest->fieldxy = malloc(mo->nfield * 2 * sizeof(double));
        memcpy(dest->fieldxy, mo->fieldxy, mo->nfield * 2 * sizeof(double));
    }
    if (mo->tagalong) {
        int i;
        dest->tagalong = bl_new(16, sizeof(tagalong_t));
        for (i=0; i<bl_size(mo->tagalong); i++) {
            tagalong_t* tag = bl_access(mo->tagalong, i);
            tagalong_t tagcopy;
            memcpy(&tagcopy, tag, sizeof(tagalong_t));
            tagcopy.name = strdup_safe(tag->name);
            tagcopy.units = strdup_safe(tag->units);
            if (tag->data) {
                tagcopy.data = malloc((size_t)tag->Ndata * (size_t)tag->itemsize);
                memcpy(tagcopy.data, tag->data, (size_t)tag->Ndata * (size_t)tag->itemsize);
            }
            bl_append(dest->tagalong, &tagcopy);
        }
    }
    // NOT SUPPORTED (yet)
    assert(!mo->field_tagalong);
}

// Free the things I added to the mo.
void onefield_free_matchobj(MatchObj* mo) {
    if (!mo) return;
    if (mo->sip) {
        sip_free(mo->sip);
        mo->sip = NULL;
    }
    free(mo->refradec);
    free(mo->fieldxy);
    free(mo->theta);
    free(mo->matchodds);
    free(mo->refxyz);
    free(mo->refxy);
    free(mo->refstarid);
    free(mo->testperm);
    mo->refradec = NULL;
    mo->fieldxy = NULL;
    mo->theta = NULL;
    mo->matchodds = NULL;
    mo->refxyz = NULL;
    mo->refxy = NULL;
    mo->refstarid = NULL;
    mo->testperm = NULL;

    if (mo->tagalong) {
        int i;
        for (i=0; i<bl_size(mo->tagalong); i++) {
            tagalong_t* tag = bl_access(mo->tagalong, i);
            free(tag->name);
            free(tag->units);
            free(tag->data);
        }
        bl_free(mo->tagalong);
        mo->tagalong = NULL;
    }
    if (mo->field_tagalong) {
        int i;
        for (i=0; i<bl_size(mo->field_tagalong); i++) {
            tagalong_t* tag = bl_access(mo->field_tagalong, i);
            free(tag->name);
            free(tag->units);
            free(tag->data);
        }
        bl_free(mo->field_tagalong);
        mo->field_tagalong = NULL;
    }
}

static void remove_duplicate_solutions(onefield_t* bp) {
    int i, j;
    // The solutions can fall out of order because tweak2() updates their logodds.
    bl_sort(bp->solutions, compare_matchobjs);

    for (i=0; i<bl_size(bp->solutions); i++) {
        MatchObj* mo = bl_access(bp->solutions, i);
        j = i+1;
        while (j < bl_size(bp->solutions)) {
            MatchObj* mo2 = bl_access(bp->solutions, j);
            if (mo->fieldfile != mo2->fieldfile)
                break;
            if (mo->fieldnum != mo2->fieldnum)
                break;
            assert(mo2->logodds <= mo->logodds);
            onefield_free_matchobj(mo2);
            verify_free_matchobj(mo2);
            bl_remove_index(bp->solutions, j);
        }
    }
}

static int write_match_file(onefield_t* bp) {
    int i;
    bp->mf = matchfile_open_for_writing(bp->matchfname);
    if (!bp->mf) {
        logerr("Failed to open file %s to write match file.\n", bp->matchfname);
        return -1;
    }
    BOILERPLATE_ADD_FITS_HEADERS(bp->mf->header);
    qfits_header_add(bp->mf->header, "DATE", qfits_get_datetime_iso8601(), "Date this file was created.", NULL);
    add_onefield_params(bp, bp->mf->header);
    if (matchfile_write_headers(bp->mf)) {
        logerr("Failed to write matchfile header.\n");
        return -1;
    }
    for (i=0; i<bl_size(bp->solutions); i++) {
        MatchObj* mo = bl_access(bp->solutions, i);
        if (matchfile_write_match(bp->mf, mo)) {
            logerr("Field %i: error writing a match.\n", mo->fieldnum);
            return -1;
        }
    }
    if (matchfile_fix_headers(bp->mf) ||
        matchfile_close(bp->mf)) {
        logerr("Error closing matchfile.\n");
        return -1;
    }
    bp->mf = NULL;
    return 0;
}

static int write_rdls_file(onefield_t* bp) {
    int i;
    qfits_header* h;
    bp->indexrdls = rdlist_open_for_writing(bp->indexrdlsfname);
    if (!bp->indexrdls) {
        logerr("Failed to open index RDLS file %s for writing.\n",
               bp->indexrdlsfname);
        return -1;
    }
    h = rdlist_get_primary_header(bp->indexrdls);

    BOILERPLATE_ADD_FITS_HEADERS(h);
    fits_add_long_history(h, "This \"indexrdls\" file contains the RA/DEC of index objects that were found inside a solved field.");
    qfits_header_add(h, "DATE", qfits_get_datetime_iso8601(), "Date this file was created.", NULL);
    add_onefield_params(bp, h);
    if (rdlist_write_primary_header(bp->indexrdls)) {
        logerr("Failed to write index RDLS header.\n");
        return -1;
    }

    for (i=0; i<bl_size(bp->solutions); i++) {
        MatchObj* mo = bl_access(bp->solutions, i);
        rd_t rd;
        if (strlen(mo->fieldname)) {
            qfits_header* hdr = rdlist_get_header(bp->indexrdls);
            qfits_header_add(hdr, "FIELDID", mo->fieldname, "Name of this field", NULL);
        }
        if (mo->tagalong) {
            int j;
            for (j=0; j<bl_size(mo->tagalong); j++) {
                tagalong_t* tag = bl_access(mo->tagalong, j);
                tag->colnum = rdlist_add_tagalong_column(bp->indexrdls, tag->type, tag->arraysize,
                                                         tag->type, tag->name, tag->units);
            }
        }
        if (rdlist_write_header(bp->indexrdls)) {
            logerr("Failed to write index RDLS field header.\n");
            return -1;
        }
        assert(mo->refradec);

        rd_from_array(&rd, mo->refradec, mo->nindex);
        if (rdlist_write_field(bp->indexrdls, &rd)) {
            logerr("Failed to write index RDLS entry.\n");
            return -1;
        }
        rd_free_data(&rd);

        if (mo->tagalong) {
            int j;
            for (j=0; j<bl_size(mo->tagalong); j++) {
                tagalong_t* tag = bl_access(mo->tagalong, j);
                if (rdlist_write_tagalong_column(bp->indexrdls, tag->colnum,
                                                 0, mo->nindex, tag->data, tag->itemsize)) {
                    ERROR("Failed to write tag-along data column %s", tag->name);
                    return -1;
                }
            }
        }

        if (rdlist_fix_header(bp->indexrdls)) {
            logerr("Failed to fix index RDLS field header.\n");
            return -1;
        }
        rdlist_next_field(bp->indexrdls);
    }

    if (rdlist_fix_primary_header(bp->indexrdls) ||
        rdlist_close(bp->indexrdls)) {
        logerr("Failed to close index RDLS file.\n");
        return -1;
    }
    bp->indexrdls = NULL;
    return 0;
}

static int write_wcs_file(onefield_t* bp) {
    int i;
    for (i=0; i<bl_size(bp->solutions); i++) {
        char wcs_fn[1024];
        FILE* fout;
        qfits_header* hdr;
        char* tm;

        MatchObj* mo = bl_access(bp->solutions, i);
        snprintf(wcs_fn, sizeof(wcs_fn), bp->wcs_template, mo->fieldnum);
        fout = fopen(wcs_fn, "wb");
        if (!fout) {
            logerr("Failed to open WCS output file %s: %s\n", wcs_fn, strerror(errno));
            return -1;
        }
        assert(mo->wcs_valid);

        if (mo->sip)
            hdr = sip_create_header(mo->sip);
        else
            hdr = tan_create_header(&(mo->wcstan));

        BOILERPLATE_ADD_FITS_HEADERS(hdr);
        qfits_header_add(hdr, "HISTORY", "This is a WCS header was created by Astrometry.net.", NULL, NULL);
        tm = qfits_get_datetime_iso8601();
        qfits_header_add(hdr, "DATE", tm, "Date this file was created.", NULL);
        add_onefield_params(bp, hdr);
        fits_add_long_comment(hdr, "-- properties of the matching quad: --");
        fits_add_long_comment(hdr, "index id: %i", mo->indexid);
        fits_add_long_comment(hdr, "index healpix: %i", mo->healpix);
        fits_add_long_comment(hdr, "index hpnside: %i", mo->hpnside);
        fits_add_long_comment(hdr, "log odds: %g", mo->logodds);
        fits_add_long_comment(hdr, "odds: %g", exp(mo->logodds));
        fits_add_long_comment(hdr, "quadno: %i", mo->quadno);
        fits_add_long_comment(hdr, "stars: %i,%i,%i,%i", mo->star[0], mo->star[1], mo->star[2], mo->star[3]);
        fits_add_long_comment(hdr, "field: %i,%i,%i,%i", mo->field[0], mo->field[1], mo->field[2], mo->field[3]);
        fits_add_long_comment(hdr, "code error: %g", sqrt(mo->code_err));
        fits_add_long_comment(hdr, "nmatch: %i", mo->nmatch);
        fits_add_long_comment(hdr, "nconflict: %i", mo->nconflict);
        fits_add_long_comment(hdr, "nfield: %i", mo->nfield);
        fits_add_long_comment(hdr, "nindex: %i", mo->nindex);
        fits_add_long_comment(hdr, "scale: %g arcsec/pix", mo->scale);
        fits_add_long_comment(hdr, "parity: %i", (int)mo->parity);
        fits_add_long_comment(hdr, "quads tried: %i", mo->quads_tried);
        fits_add_long_comment(hdr, "quads matched: %i", mo->quads_matched);
        fits_add_long_comment(hdr, "quads verified: %i", mo->nverified);
        fits_add_long_comment(hdr, "objs tried: %i", mo->objs_tried);
        fits_add_long_comment(hdr, "cpu time: %g", mo->timeused);
        fits_add_long_comment(hdr, "--");

        if (strlen(mo->fieldname))
            qfits_header_add(hdr, bp->fieldid_key, mo->fieldname, "Field name (copied from input field)", NULL);
			
        if (qfits_header_dump(hdr, fout)) {
            logerr("Failed to write FITS WCS header.\n");
            return -1;
        }
        fits_pad_file(fout);
        qfits_header_destroy(hdr);
        fclose(fout);
    }
    return 0;
}

static int write_scamp_file(onefield_t* bp) {
    int i;
    scamp_cat_t* scamp;
    qfits_header* hdr = NULL;
    MatchObj* mo;
    tan_t fakewcs;

    // HACK -- just hdr = NULL?
    hdr = qfits_header_default();
    fits_header_add_int(hdr, "BITPIX", 0, NULL);
    fits_header_add_int(hdr, "NAXIS", 2, NULL);
    fits_header_add_int(hdr, "NAXIS1", 0, NULL);
    fits_header_add_int(hdr, "NAXIS2", 0, NULL);
    qfits_header_add(hdr, "EXTEND", "T", "", NULL);
    memset(&fakewcs, 0, sizeof(tan_t));
    tan_add_to_header(hdr, &fakewcs);

    scamp = scamp_catalog_open_for_writing(bp->scamp_fname, TRUE);
    if (!scamp) {
        logerr("Failed to open SCAMP reference catalog for writing.\n");
        return -1;
    }
    if (scamp_catalog_write_field_header(scamp, hdr)) {
        logerr("Failed to write SCAMP headers.\n");
        return -1;
    }
    mo = bl_access(bp->solutions, 0);
    for (i=0; i<mo->nindex; i++) {
        scamp_ref_t ref;
        ref.ra  = mo->refradec[2*i + 0];
        ref.dec = mo->refradec[2*i + 1];
        ref.err_a = ref.err_b = arcsec2deg(mo->index_jitter);
        // HACK
        ref.mag = 10.0;
        ref.err_mag = 0.1;

        if (scamp_catalog_write_reference(scamp, &ref)) {
            logerr("Failed to write SCAMP object.\n");
            return -1;
        }
    }
    if (scamp_catalog_close(scamp)) {
        logerr("Failed to close SCAMP reference catalog.\n");
        return -1;
    }
    return 0;
}

static int write_corr_file(onefield_t* bp) {
    int i;
    fitstable_t* tab;
    tab = fitstable_open_for_writing(bp->corr_fname);
    if (!tab) {
        ERROR("Failed to open correspondences file \"%s\" for writing", bp->corr_fname);
        return -1;
    }
    // FIXME -- add header boilerplate.

    if (fitstable_write_primary_header(tab)) {
        ERROR("Failed to write primary header for corr file \"%s\"", bp->corr_fname);
        return -1;
    }

    for (i=0; i<bl_size(bp->solutions); i++) {
        MatchObj* mo;
        sip_t thesip;
        sip_t* wcs;
        int j;
        tfits_type dubl = fitscolumn_double_type();
        tfits_type itype = fitscolumn_int_type();

        mo = bl_access(bp->solutions, i);

        if (mo->sip)
            wcs = mo->sip;
        else {
            sip_wrap_tan(&mo->wcstan, &thesip);
            wcs = &thesip;
        }

        fitstable_add_write_column(tab, dubl, "field_x",   "pixels");
        fitstable_add_write_column(tab, dubl, "field_y",   "pixels");
        fitstable_add_write_column(tab, dubl, "field_ra",  "degrees");
        fitstable_add_write_column(tab, dubl, "field_dec", "degrees");
        fitstable_add_write_column(tab, dubl, "index_x",   "pixels");
        fitstable_add_write_column(tab, dubl, "index_y",   "pixels");
        fitstable_add_write_column(tab, dubl, "index_ra",  "degrees");
        fitstable_add_write_column(tab, dubl, "index_dec", "degrees");
        fitstable_add_write_column(tab, itype, "index_id", "none");
        fitstable_add_write_column(tab, itype, "field_id", "none");
        fitstable_add_write_column(tab, dubl, "match_weight", "none");
		
        if (mo->tagalong) {
            for (j=0; j<bl_size(mo->tagalong); j++) {
                tagalong_t* tag = bl_access(mo->tagalong, j);
                fitstable_add_write_column_struct(tab, tag->type, tag->arraysize, 0, tag->type, tag->name, tag->units);
                tag->colnum = fitstable_ncols(tab)-1;
            }
        }

        // FIXME -- check for duplicate column names
        if (mo->field_tagalong) {
            int j;
            for (j=0; j<bl_size(mo->field_tagalong); j++) {
                tagalong_t* tag = bl_access(mo->field_tagalong, j);
                fitstable_add_write_column_struct(tab, tag->type, tag->arraysize, 0, tag->type, tag->name, tag->units);
                tag->colnum = fitstable_ncols(tab)-1;
            }
        }

        if (fitstable_write_header(tab)) {
            ERROR("Failed to write correspondence file header.");
            return -1;
        }

        {
            int rows = 0;
            for (j=0; j<mo->nfield; j++) {
                if (mo->theta[j] < 0)
                    continue;
                rows++;
            }
            logverb("Writing %i rows (of %i field and %i index objects) to correspondence file.\n", rows, mo->nfield, mo->nindex);
        }
        for (j=0; j<mo->nfield; j++) {
            double fx,fy,fra,fdec;
            double rx,ry,rra,rdec;
            double weight;
            int ti, ri;
            ri = mo->theta[j];
            if (ri < 0)
                continue;
            ti = j;
            rra  = mo->refradec[2*ri+0];
            rdec = mo->refradec[2*ri+1];
            if (!sip_radec2pixelxy(wcs, rra, rdec, &rx, &ry))
                continue;
            fx = mo->fieldxy[2*ti+0];
            fy = mo->fieldxy[2*ti+1];
            sip_pixelxy2radec(wcs, fx, fy, &fra, &fdec);
            logdebug("Writing field xy %.1f,%.1f, radec %.2f,%.2f; index xy %.1f,%.1f, radec %.2f,%.2f\n", fx, fy, fra, fdec, rx, ry, rra, rdec);
            weight = verify_logodds_to_weight(mo->matchodds[j]);
            if (fitstable_write_row(tab, &fx, &fy, &fra, &fdec, &rx, &ry, &rra, &rdec, &ri, &ti, &weight)) {
                ERROR("Failed to write coordinates to correspondences file \"%s\"", bp->corr_fname);
                return -1;
            }
        }

        if (mo->tagalong) {
            for (j=0; j<bl_size(mo->tagalong); j++) {
                tagalong_t* tag = bl_access(mo->tagalong, j);
                int row = 0;
                int k;
                // Ugh, we write each datum individually...
                for (k=0; k<mo->nfield; k++) {
                    int ri = mo->theta[k];
                    if (ri < 0)
                        continue;
                    fitstable_write_one_column(tab, tag->colnum, row, 1,
                                               (char*)tag->data + ri*tag->itemsize, 0);
                    row++;
                }
            }
        }
        if (mo->field_tagalong) {
            for (j=0; j<bl_size(mo->field_tagalong); j++) {
                tagalong_t* tag = bl_access(mo->field_tagalong, j);
                int row = 0;
                int k;
                // Ugh, we write each datum individually...
                for (k=0; k<mo->nfield; k++) {
                    if (mo->theta[k] < 0)
                        continue;
                    fitstable_write_one_column(tab, tag->colnum, row, 1,
                                               (char*)tag->data + k*tag->itemsize, 0);
                    row++;
                }
            }
        }
		
        if (fitstable_fix_header(tab)) {
            ERROR("Failed to fix correspondence file header.");
            return -1;
        }

        fitstable_next_extension(tab);
        fitstable_clear_table(tab);
    }

    if (fitstable_close(tab)) {
        ERROR("Failed to close correspondence file");
        return -1;
    }

    return 0;
}

static int write_solutions(onefield_t* bp) {
    anbool got_solutions = (bl_size(bp->solutions) > 0);

    // If we found no solution, don't write empty output files!
    if (!got_solutions)
        return 0;

    // The solutions can fall out of order because tweak2() updates their logodds.
    bl_sort(bp->solutions, compare_matchobjs);

    if (bp->matchfname) {
        if (write_match_file(bp))
            return -1;
    }
    if (bp->indexrdlsfname) {
        if (write_rdls_file(bp))
            return -1;
    }

    // We only want the best solution for each field in the following outputs:
    remove_duplicate_solutions(bp);

    if (bp->wcs_template) {
        if (write_wcs_file(bp))
            return -1;
    }
    if (bp->scamp_fname) {
        if (write_scamp_file(bp))
            return -1;
    }
    if (bp->corr_fname) {
        if (write_corr_file(bp))
            return -1;
    }
    return 0;
}

static int compare_matchobjs(const void* v1, const void* v2) {
    int diff;
    float fdiff;
    const MatchObj* mo1 = v1;
    const MatchObj* mo2 = v2;
    diff = mo1->fieldfile - mo2->fieldfile;
    if (diff) return diff;
    diff = mo1->fieldnum - mo2->fieldnum;
    if (diff) return diff;
    fdiff = mo1->logodds - mo2->logodds;
    if (fdiff == 0.0)
        return 0;
    if (fdiff > 0.0)
        return -1;
    return 1;
}
