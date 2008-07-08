/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

/**
 *   Solve fields blindly
 *
 * Inputs: .ckdt .quad .skdt
 * Output: .match .rdls .wcs
 */

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/param.h>
#include <libgen.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "blind.h"
#include "tweak.h"
#include "sip_qfits.h"
#include "starutil.h"
#include "mathutil.h"
#include "quadfile.h"
#include "solvedclient.h"
#include "solvedfile.h"
#include "starkd.h"
#include "codekd.h"
#include "boilerplate.h"
#include "fitsioutils.h"
#include "blind_wcs.h"
#include "verify.h"
#include "index.h"
#include "log.h"
#include "tic.h"
#include "qfits_table.h"
#include "errors.h"

static bool record_match_callback(MatchObj* mo, void* userdata);
static time_t timer_callback(void* user_data);
static void add_blind_params(blind_t* bp, qfits_header* hdr);
static void get_fields_from_solvedserver(blind_t* bp, solver_t* sp);
static void load_and_parse_wcsfiles(blind_t* bp);
static void solve_fields(blind_t* bp, sip_t* verify_wcs);
static void remove_invalid_fields(il* fieldlist, int maxfield);
static bool is_field_solved(blind_t* bp, int fieldnum);
static int write_solutions(blind_t* bp);
static void solved_field(blind_t* bp, int fieldnum);
static int compare_matchobjs(const void* v1, const void* v2);
static void remove_duplicate_solutions(blind_t* bp);
static void free_matchobj(MatchObj* mo);

/** Index handling for in_parallel and not.

 Currently it supposedly could handle both "indexnames" and "indexes",
 but we should probably just assert that only one of these can be used.
 **/
static index_t* get_index(blind_t* bp, int i) {
    if (i < sl_size(bp->indexnames)) {
        char* fn = sl_get(bp->indexnames, i);
        index_t* ind = index_load(fn, bp->index_options);
        if (!ind) {
            ERROR("Failed to load index %s", fn);
            exit( -1);
        }
        return ind;
    }
    i -= sl_size(bp->indexnames);
    return pl_get(bp->indexes, i);
}
static char* get_index_name(blind_t* bp, int i) {
    index_t* index;
    if (i < sl_size(bp->indexnames)) {
        char* fn = sl_get(bp->indexnames, i);
        return fn;
    }
    i -= sl_size(bp->indexnames);
    index = pl_get(bp->indexes, i);
    return index->meta.indexname;
}
static void done_with_index(blind_t* bp, int i, index_t* ind) {
    if (i < sl_size(bp->indexnames)) {
        index_close(ind);
    }
}
static int n_indexes(blind_t* bp) {
    return sl_size(bp->indexnames) + pl_size(bp->indexes);
}



void blind_clear_verify_wcses(blind_t* bp) {
    bl_remove_all(bp->verify_wcs_list);
}

void blind_clear_solutions(blind_t* bp) {
	bl_remove_all(bp->solutions);
}

void blind_clear_indexes(blind_t* bp) {
    sl_remove_all(bp->indexnames);
}

void blind_set_field_file(blind_t* bp, const char* fn) {
    free(bp->fieldfname);
    bp->fieldfname = strdup_safe(fn);
}

void blind_set_solved_file(blind_t* bp, const char* fn) {
    blind_set_solvedin_file (bp, fn);
    blind_set_solvedout_file(bp, fn);
}

void blind_set_solvedin_file(blind_t* bp, const char* fn) {
    free(bp->solved_in);
    bp->solved_in = strdup_safe(fn);
}

void blind_set_solvedout_file(blind_t* bp, const char* fn) {
    free(bp->solved_out);
    bp->solved_out = strdup_safe(fn);
}

void blind_set_cancel_file(blind_t* bp, const char* fn) {
    free(bp->cancelfname);
    bp->cancelfname = strdup_safe(fn);
}

void blind_set_match_file(blind_t* bp, const char* fn) {
    free(bp->matchfname);
    bp->matchfname = strdup_safe(fn);
}

void blind_set_rdls_file(blind_t* bp, const char* fn) {
    free(bp->indexrdlsfname);
    bp->indexrdlsfname = strdup_safe(fn);
}

void blind_set_corr_file(blind_t* bp, const char* fn) {
    free(bp->corr_fname);
    bp->corr_fname = strdup_safe(fn);
}

void blind_set_wcs_file(blind_t* bp, const char* fn) {
    free(bp->wcs_template);
    bp->wcs_template = strdup_safe(fn);
}

void blind_set_xcol(blind_t* bp, const char* x) {
    free(bp->xcolname);
    if (!x)
        x = "X";
    bp->xcolname = strdup(x);
}

void blind_set_ycol(blind_t* bp, const char* y) {
    free(bp->ycolname);
    if (!y)
        y = "Y";
    bp->ycolname = strdup_safe(y);
}

void blind_add_index(blind_t* bp, const char* index) {
    sl_append(bp->indexnames, index);
}

void blind_add_loaded_index(blind_t* bp, index_t* ind) {
    pl_append(bp->indexes, ind);
}

void blind_add_verify_wcs(blind_t* bp, sip_t* wcs) {
    bl_append(bp->verify_wcs_list, wcs);
}

void blind_add_field(blind_t* bp, int field) {
    il_insert_unique_ascending(bp->fieldlist, field);
}

void blind_add_field_range(blind_t* bp, int lo, int hi) {
    int i;
    for (i=lo; i<=hi; i++) {
        il_insert_unique_ascending(bp->fieldlist, i);
    }
}

static void check_time_limits(blind_t* bp) {
	if (bp->total_timelimit || bp->timelimit) {
		time_t now = time(NULL);
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
		float now = get_cpu_usage(bp);
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

void blind_run(blind_t* bp) {
	solver_t* sp = &(bp->solver);
	int i, I;
    int Nindexes;

	// Record current time for total wall-clock time limit.
	bp->time_total_start = time(NULL);

	// Record current CPU usage for total cpu-usage limit.
	bp->cpu_total_start = get_cpu_usage(bp);

	get_fields_from_solvedserver(bp, sp);

	// Parse WCS files submitted for verification.
	load_and_parse_wcsfiles(bp);

	// Read .xyls file...
	logverb("Reading fields file %s...", bp->fieldfname);
	bp->xyls = xylist_open(bp->fieldfname);
	if (!bp->xyls) {
		logerr("Failed to read xylist.\n");
		exit( -1);
	}
	xylist_set_xname(bp->xyls, bp->xcolname);
	xylist_set_yname(bp->xyls, bp->ycolname);
    xylist_set_include_flux(bp->xyls, FALSE);
    xylist_set_include_background(bp->xyls, FALSE);
	logverb("found %u fields.\n", xylist_n_fields(bp->xyls));

    remove_invalid_fields(bp->fieldlist, xylist_n_fields(bp->xyls));

    if (bp->use_idfile)
        bp->index_options |= INDEX_USE_IDS;

    Nindexes = n_indexes(bp);

	// Verify any WCS estimates we have.
	if (bl_size(bp->verify_wcs_list)) {
        int i;
		int w;

        // We want to get the best logodds out of all the indices, so we set the
        // logodds-to-solve impossibly high so that a "good enough" solution doesn't
        // stop us from continuing to search...
        double oldodds = bp->logratio_tosolve;
        bp->logratio_tosolve = HUGE_VAL;

		for (w = 0; w < bl_size(bp->verify_wcs_list); w++) {
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
			quadlo = bp->quad_size_fraction_lo * MIN(wcs->wcstan.imagew, wcs->wcstan.imageh) * sip_pixel_scale(wcs);
			quadhi = bp->quad_size_fraction_hi * MAX(wcs->wcstan.imagew, wcs->wcstan.imageh) * sip_pixel_scale(wcs);
			logmsg("Verifying WCS using indices with quads of size [%g, %g] arcmin\n",
				   arcsec2arcmin(quadlo), arcsec2arcmin(quadhi));

			for (I=0; I<Nindexes; I++) {
                index_t* index = get_index(bp, I);
                if ((index->meta.index_scale_lower > quadhi) ||
                    (index->meta.index_scale_upper < quadlo)) {
                    done_with_index(bp, I, index);
					continue;
				}
                solver_add_index(sp, index);
				sp->index = index;
				logmsg("Verifying WCS with index %i of %i\n",  I + 1, sl_size(bp->indexnames));
				// Do it!
				solve_fields(bp, wcs);
				// Clean up this index...
                done_with_index(bp, I, index);
                solver_clear_indexes(sp);
			}
		}

        bp->logratio_tosolve = oldodds;

		logmsg("Got %i solutions.\n", bl_size(bp->solutions));

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
		bp->cpu_start = get_cpu_usage(bp);
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
			logverb("Trying index %s...\n", index->meta.indexname);

			// Record current CPU usage.
			bp->cpu_start = get_cpu_usage(bp);
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

	if (bp->solvedserver)
		solvedclient_set_server(NULL);

    if (write_solutions(bp))
        exit(-1);

	for (i=0; i<bl_size(bp->solutions); i++) {
		MatchObj* mo = bl_access(bp->solutions, i);
		free_matchobj(mo);
	}
	bl_remove_all(bp->solutions);
}

void blind_init(blind_t* bp) {
	// Reset params.
	memset(bp, 0, sizeof(blind_t));

	bp->fieldlist = il_new(256);
    bp->solutions = bl_new(16, sizeof(MatchObj));
	bp->indexnames = sl_new(16);
	bp->indexes = pl_new(16);
	bp->verify_wcs_list = bl_new(1, sizeof(sip_t));
	bp->verify_wcsfiles = sl_new(1);
	bp->fieldid_key = strdup("FIELDID");
    blind_set_xcol(bp, NULL);
    blind_set_ycol(bp, NULL);
	bp->firstfield = -1;
	bp->lastfield = -1;
	bp->tweak_aborder = DEFAULT_TWEAK_ABORDER;
	bp->tweak_abporder = DEFAULT_TWEAK_ABPORDER;
    bp->quad_size_fraction_lo = DEFAULT_QSF_LO;
    bp->quad_size_fraction_hi = DEFAULT_QSF_HI;
    bp->nsolves = 1;
    // don't set sp-> here because solver_set_default_values()
    // will get called next and wipe it out...
}

int blind_parameters_are_sane(blind_t* bp, solver_t* sp) {
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
	if ((((sp->verify_pix > 0.0) ? 1 : 0) +
		 ((bp->verify_dist2 > 0.0) ? 1 : 0)) != 1) {
		logerr("You must specify either verify_pix or verify_dist2.\n");
		return 0;
	}
	if (bp->verify_dist2 > 0.0) {
		logerr("verify_dist2 mode is broken; email mierle@gmail.com to complain.\n");
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

int blind_is_run_obsolete(blind_t* bp, solver_t* sp) {
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

static void get_fields_from_solvedserver(blind_t* bp, solver_t* sp) {
	if (!bp->solvedserver)
        return;
    if (solvedclient_set_server(bp->solvedserver)) {
        logerr("Error setting solvedserver.\n");
        exit( -1);
    }

    if ((il_size(bp->fieldlist) == 0) && (bp->firstfield != -1) && (bp->lastfield != -1)) {
        int j;
        il_free(bp->fieldlist);
        logmsg("Contacting solvedserver to get field list...\n");
        bp->fieldlist = solvedclient_get_fields(bp->fieldid, bp->firstfield, bp->lastfield, 0);
        if (!bp->fieldlist) {
            logerr("Failed to get field list from solvedserver.\n");
            exit( -1);
        }
        logmsg("Got %i fields from solvedserver: ", il_size(bp->fieldlist));
        for (j = 0; j < il_size(bp->fieldlist); j++) {
            logmsg("%i ", il_get(bp->fieldlist, j));
        }
        logmsg("\n");
    }
}

static void load_and_parse_wcsfiles(blind_t* bp) {
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

void blind_log_run_parameters(blind_t* bp) {
	solver_t* sp = &(bp->solver);
	int i, N;

    logverb("blind solver run parameters:\n");
	logverb("fields ");
	for (i = 0; i < il_size(bp->fieldlist); i++)
		logverb("%i ", il_get(bp->fieldlist, i));
	logverb("\n");
	logverb("indexes:\n");
    N = n_indexes(bp);
    for (i=0; i<N; i++)
        logverb("  %s\n", get_index_name(bp, i));
	logverb("fieldfname %s\n", bp->fieldfname);
	for (i = 0; i < sl_size(bp->verify_wcsfiles); i++)
		logverb("verify %s\n", sl_get(bp->verify_wcsfiles, i));
	logverb("fieldid %i\n", bp->fieldid);
	logverb("matchfname %s\n", bp->matchfname);
	logverb("solved_in %s\n", bp->solved_in);
	logverb("solved_out %s\n", bp->solved_out);
	logverb("solvedserver %s\n", bp->solvedserver);
	logverb("cancel %s\n", bp->cancelfname);
	logverb("wcs %s\n", bp->wcs_template);
	logverb("fieldid_key %s\n", bp->fieldid_key);
	logverb("parity %i\n", sp->parity);
	logverb("codetol %g\n", sp->codetol);
	logverb("startdepth %i\n", sp->startobj);
	logverb("enddepth %i\n", sp->endobj);
	logverb("fieldunits_lower %g\n", sp->funits_lower);
	logverb("fieldunits_upper %g\n", sp->funits_upper);
	logverb("verify_dist %g\n", distsq2arcsec(bp->verify_dist2));
	logverb("verify_pix %g\n", sp->verify_pix);
	logverb("xcolname %s\n", bp->xcolname);
	logverb("ycolname %s\n", bp->ycolname);
	logverb("maxquads %i\n", sp->maxquads);
	logverb("maxmatches %i\n", sp->maxmatches);
	logverb("cpulimit %f\n", bp->cpulimit);
	logverb("timelimit %i\n", bp->timelimit);
	logverb("total_timelimit %i\n", bp->total_timelimit);
	logverb("total_cpulimit %f\n", bp->total_cpulimit);
	logverb("tweak %s\n", bp->do_tweak ? "on" : "off");
	if (bp->do_tweak) {
		logverb("tweak_aborder %i\n", bp->tweak_aborder);
		logverb("tweak_abporder %i\n", bp->tweak_abporder);
	}
}

void blind_cleanup(blind_t* bp) {
	il_free(bp->fieldlist);
    bl_free(bp->solutions);
	sl_free2(bp->indexnames);
	pl_free(bp->indexes);
	sl_free2(bp->verify_wcsfiles);
	bl_free(bp->verify_wcs_list);

	free(bp->cancelfname);
	free(bp->fieldfname);
	free(bp->fieldid_key);
	free(bp->indexrdlsfname);
	free(bp->corr_fname);
	free(bp->matchfname);
	free(bp->solvedserver);
	free(bp->solved_in);
	free(bp->solved_out);
	free(bp->wcs_template);
	free(bp->xcolname);
	free(bp->ycolname);
}

static sip_t* tweak(blind_t* bp, tan_t* wcs, const double* starradec, int nstars) {
	solver_t* sp = &(bp->solver);
	tweak_t* twee = NULL;
	sip_t* sip = NULL;

	logmsg("Tweaking!\n");

	twee = tweak_new();

	if (bp->verify_dist2 > 0.0)
		twee->jitter = distsq2arcsec(bp->verify_dist2);
	else {
		twee->jitter = hypot(tan_pixel_scale(wcs) * sp->verify_pix, sp->index->meta.index_jitter);
		logverb("Star jitter: %g arcsec.\n", twee->jitter);
	}
	logverb("Setting tweak jitter: %g arcsec.\n", twee->jitter);

	twee->sip->a_order  = twee->sip->b_order  = bp->tweak_aborder;
	twee->sip->ap_order = twee->sip->bp_order = bp->tweak_abporder;
	twee->weighted_fit = TRUE;

    // Image coords:
	logverb("Tweaking using %i image coordinates.\n", starxy_n(sp->fieldxy));
    tweak_push_image_xy(twee, sp->fieldxy);

    // Index coords:
	logverb("Tweaking using %i star coordinates.\n", nstars);
	tweak_push_ref_ad_array(twee, starradec, nstars);

	tweak_push_wcs_tan(twee, wcs);

	if (bp->tweak_skipshift) {
		logverb("Skipping shift operation.\n");
		tweak_skip_shift(twee);
	}

	logverb("Begin tweaking (to order %i)...\n", bp->tweak_aborder);
    tweak_iterate_to_order(twee, MAX(1, bp->tweak_aborder), 5);
	logverb("Done tweaking!\n");

	// Steal the resulting SIP structure
	sip = twee->sip;
	twee->sip = NULL;

	tweak_free(twee);
	return sip;
}

static void print_match(blind_t* bp, MatchObj* mo) {
	int Nmin = MIN(mo->nindex, mo->nfield);
	int ndropout = Nmin - mo->noverlap - mo->nconflict;
	logverb("  logodds ratio %g (%g), %i match, %i conflict, %i dropout, %i index.\n",
	       mo->logodds, exp(mo->logodds), mo->noverlap, mo->nconflict, ndropout, mo->nindex);
}

static bool record_match_callback(MatchObj* mo, void* userdata) {
	blind_t* bp = userdata;
	solver_t* sp = &(bp->solver);
    MatchObj* ourmo;
    int ind;
    int j;

	check_time_limits(bp);

	if (mo->logodds >= bp->logratio_toprint)
        print_match(bp, mo);

	if (mo->logodds < bp->logratio_tokeep)
		return FALSE;

    logverb("Pixel scale: %g arcsec/pix.\n", mo->scale);

    if (bp->do_tweak || bp->indexrdlsfname) {
        // Gather stars that are within range.
        double* radec = NULL;
        int nstars;
        double rad2, safety;

        // add a small margin.
        safety = 1.05;
        rad2 = square(safety * mo->radius);

        startree_search(sp->index->starkd, mo->center, rad2, NULL, &radec, &nstars);

        if (bp->do_tweak)
            mo->sip = tweak(bp, &(mo->wcstan), radec, nstars);

        if (bp->indexrdlsfname) {
            // steal this array...
            mo->indexrdls = radec;
            radec = NULL;
            mo->nindexrdls = nstars;
        }

        free(radec);
    }

    ind = bl_insert_sorted(bp->solutions, mo, compare_matchobjs);
    ourmo = bl_access(bp->solutions, ind);

    ourmo->corr_field = NULL;
    ourmo->corr_index = NULL;
    ourmo->corr_field_xy = dl_new(16);
    ourmo->corr_index_rd = dl_new(16);

    for (j=0; j<il_size(mo->corr_field); j++) {
        double ixyz[3];
        double iradec[2];
        int iindex, ifield;

        ifield = il_get(mo->corr_field, j);
        iindex = il_get(mo->corr_index, j);
        assert(ifield >= 0);
        assert(ifield < starxy_n(sp->fieldxy));

        dl_append(ourmo->corr_field_xy,
                  starxy_getx(sp->fieldxy, ifield));
        dl_append(ourmo->corr_field_xy,
                  starxy_gety(sp->fieldxy, ifield));

        startree_get(sp->index->starkd, iindex, ixyz);
        xyzarr2radecdegarr(ixyz, iradec);

        dl_append(ourmo->corr_index_rd, iradec[0]);
        dl_append(ourmo->corr_index_rd, iradec[1]);
    }

	if (mo->logodds < bp->logratio_tosolve)
		return FALSE;

    bp->nsolves_sofar++;
    if (bp->nsolves_sofar >= bp->nsolves) {
        if (bp->solver.index) {
            char* copy;
            char* base;
            copy = strdup(bp->solver.index->meta.indexname);
            base = strdup(basename(copy));
            free(copy);
            logmsg("Field %i: solved with index %s.\n", mo->fieldnum, base);
            free(base);
        } else {
            logmsg("Field %i: solved with index %i", mo->fieldnum, mo->indexid);
            if (mo->healpix >= 0)
                logmsg(", healpix %i\n", mo->healpix);
            else
                logmsg("\n");
        }
        return TRUE;
    } else {
        logmsg("Found a quad that solves the image; that makes %i of %i required.\n",
               bp->nsolves_sofar, bp->nsolves);
    }
	return FALSE;
}

static time_t timer_callback(void* user_data) {
	blind_t* bp = user_data;

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

static void add_blind_params(blind_t* bp, qfits_header* hdr) {
	solver_t* sp = &(bp->solver);
	int i;
    int Nindexes;
	fits_add_long_comment(hdr, "-- blind solver parameters: --");
	if (sp->index) {
		fits_add_long_comment(hdr, "Index name: %s", sp->index->meta.indexname);
		fits_add_long_comment(hdr, "Index id: %i", sp->index->meta.indexid);
		fits_add_long_comment(hdr, "Index healpix: %i", sp->index->meta.healpix);
		fits_add_long_comment(hdr, "Index healpix nside: %i", sp->index->meta.hpnside);
		fits_add_long_comment(hdr, "Index scale lower: %g arcsec", sp->index->meta.index_scale_lower);
		fits_add_long_comment(hdr, "Index scale upper: %g arcsec", sp->index->meta.index_scale_upper);
		fits_add_long_comment(hdr, "Index jitter: %g", sp->index->meta.index_jitter);
		fits_add_long_comment(hdr, "Circle: %s", sp->index->meta.circle ? "yes" : "no");
		fits_add_long_comment(hdr, "Cxdx margin: %g", sp->cxdx_margin);
	}
    Nindexes = n_indexes(bp);
	for (i = 0; i < Nindexes; i++)
		fits_add_long_comment(hdr, "Index(%i): %s", i, get_index_name(bp, i));

	fits_add_long_comment(hdr, "Field name: %s", bp->fieldfname);
	fits_add_long_comment(hdr, "Field scale lower: %g arcsec/pixel", sp->funits_lower);
	fits_add_long_comment(hdr, "Field scale upper: %g arcsec/pixel", sp->funits_upper);
	fits_add_long_comment(hdr, "X col name: %s", bp->xcolname);
	fits_add_long_comment(hdr, "Y col name: %s", bp->ycolname);
	fits_add_long_comment(hdr, "Start obj: %i", sp->startobj);
	fits_add_long_comment(hdr, "End obj: %i", sp->endobj);

	fits_add_long_comment(hdr, "Solved_in: %s", bp->solved_in);
	fits_add_long_comment(hdr, "Solved_out: %s", bp->solved_out);
	fits_add_long_comment(hdr, "Solvedserver: %s", bp->solvedserver);

	fits_add_long_comment(hdr, "Parity: %i", sp->parity);
	fits_add_long_comment(hdr, "Codetol: %g", sp->codetol);
	fits_add_long_comment(hdr, "Verify distance: %g arcsec", distsq2arcsec(bp->verify_dist2));
	fits_add_long_comment(hdr, "Verify pixels: %g pix", sp->verify_pix);

	fits_add_long_comment(hdr, "Maxquads: %i", sp->maxquads);
	fits_add_long_comment(hdr, "Maxmatches: %i", sp->maxmatches);
	fits_add_long_comment(hdr, "Cpu limit: %f s", bp->cpulimit);
	fits_add_long_comment(hdr, "Time limit: %i s", bp->timelimit);
	fits_add_long_comment(hdr, "Total time limit: %i s", bp->total_timelimit);
	fits_add_long_comment(hdr, "Total CPU limit: %f s", bp->total_cpulimit);

	fits_add_long_comment(hdr, "Tweak: %s", (bp->do_tweak ? "yes" : "no"));
	if (bp->do_tweak) {
		fits_add_long_comment(hdr, "Tweak AB order: %i", bp->tweak_aborder);
		fits_add_long_comment(hdr, "Tweak ABP order: %i", bp->tweak_abporder);
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

static void solve_fields(blind_t* bp, sip_t* verify_wcs) {
	solver_t* sp = &(bp->solver);
	double last_utime, last_stime;
	double utime, stime;
	struct timeval wtime, last_wtime;
	int nfields;
	int fi;

	get_resource_stats(&last_utime, &last_stime, NULL);
	gettimeofday(&last_wtime, NULL);

	nfields = xylist_n_fields(bp->xyls);

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
			char* idstr = qfits_pretty_string(qfits_header_getstr(fieldhdr, bp->fieldid_key));
			if (idstr)
				strncpy(template.fieldname, idstr, sizeof(template.fieldname) - 1);
		}

		// Has the field already been solved?
        if (is_field_solved(bp, fieldnum))
            goto cleanup;

		// Get the field.
        sp->fieldxy = xylist_read_field(bp->xyls, NULL);
        if (!sp->fieldxy) {
            logerr("Failed to read xylist field.\n");
            goto cleanup;
        }

		sp->numtries = 0;
		sp->nummatches = 0;
		sp->numscaleok = 0;
		sp->num_cxdx_skipped = 0;
		sp->num_verified = 0;
		sp->quit_now = FALSE;
		sp->mo_template = &template ;

		solver_reset_best_match(sp);

		bp->fieldnum = fieldnum;
        bp->nsolves_sofar = 0;

		sp->logratio_record_threshold = MIN(bp->logratio_tokeep, bp->logratio_toprint);
		sp->record_match_callback = record_match_callback;
        sp->timer_callback = timer_callback;
		sp->userdata = bp;

		solver_preprocess_field(sp);

		if (verify_wcs) {
			// fabricate a match...
			MatchObj mo;
			memcpy(&mo, &template, sizeof(MatchObj));
			memcpy(&(mo.wcstan), &(verify_wcs->wcstan), sizeof(tan_t));
			mo.wcs_valid = TRUE;
			mo.scale = sip_pixel_scale(verify_wcs);

			//solver_transform_corners(sp, &mo);
            sip_pixelxy2xyzarr(verify_wcs, sp->field_minx, sp->field_miny, mo.sMin);
            sip_pixelxy2xyzarr(verify_wcs, sp->field_maxx, sp->field_maxy, mo.sMax);
            sip_pixelxy2xyzarr(verify_wcs, sp->field_minx, sp->field_maxy, mo.sMinMax);
            sip_pixelxy2xyzarr(verify_wcs, sp->field_maxx, sp->field_miny, mo.sMaxMin);
            star_midpoint(mo.center, mo.sMin, mo.sMax);
            mo.radius = sqrt(distsq(mo.center, mo.sMin, 3));

			sp->distance_from_quad_bonus = FALSE;

			logmsg("Verifying WCS of field %i.\n", fieldnum);

			solver_inject_match(sp, &mo, verify_wcs);

			// print it, if it hasn't been already.
			if (mo.logodds < bp->logratio_toprint)
				print_match(bp, &mo);

            // HACK
            il_free(mo.corr_field);
            il_free(mo.corr_index);

		} else {
			logverb("Solving field %i.\n", fieldnum);

			sp->distance_from_quad_bonus = TRUE;
			
			// The real thing
			solver_run(sp);

			logverb("Field %i: tried %i quads, matched %i codes.\n",
                    fieldnum, sp->numtries, sp->nummatches);

			if (sp->maxquads && sp->numtries >= sp->maxquads) {
				logmsg("  exceeded the number of quads to try: %i >= %i.\n",
				       sp->numtries, sp->maxquads);
			}
			if (sp->maxmatches && sp->nummatches >= sp->maxmatches) {
				logmsg("  exceeded the number of quads to match: %i >= %i.\n",
				       sp->nummatches, sp->maxmatches);
			}
			if (bp->cancelled) {
				logmsg("  cancelled at user request.\n");
			}
		}


        if (sp->best_match_solves) {
            solved_field(bp, fieldnum);
		} else if (!verify_wcs) {
			// Field unsolved.
            logerr("Field %i did not solve", fieldnum);
            if (bp->solver.index && bp->solver.index->meta.indexname) {
                char* copy;
                char* base;
                copy = strdup(bp->solver.index->meta.indexname);
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
				print_match(bp, &(sp->best_match));
			} else {
				logverb("Best odds encountered: %g\n", exp(sp->best_logodds));
			}
		}

		solver_free_field(sp);

		get_resource_stats(&utime, &stime, NULL);
		gettimeofday(&wtime, NULL);
		logverb("Spent %g s user, %g s system, %g s total, %g s wall time.\n",
		       (utime - last_utime), (stime - last_stime), (stime - last_stime + utime - last_utime),
		       millis_between(&last_wtime, &wtime) * 0.001);

		last_utime = utime;
		last_stime = stime;
		last_wtime = wtime;

	cleanup:
        starxy_free(sp->fieldxy);
	}

    sp->fieldxy = NULL;
}

static bool is_field_solved(blind_t* bp, int fieldnum) {
  bool solved = FALSE;
    if (bp->solved_in) {
      solved = solvedfile_get(bp->solved_in, fieldnum);
      logverb("Checking %s field %i to see if the field is solved: %s.\n",
	      bp->solved_in, fieldnum, (solved ? "yes" : "no"));
    }
    if (solved) {
        // file exists; field has already been solved.
        logmsg("Field %i: solvedfile %s: field has been solved.\n", fieldnum, bp->solved_in);
        return TRUE;
    }
    if (bp->solvedserver &&
        (solvedclient_get(bp->fieldid, fieldnum) == 1)) {
        // field has already been solved.
        logmsg("Field %i: field has already been solved.\n", fieldnum);
        return TRUE;
    }
    return FALSE;
}

static void solved_field(blind_t* bp, int fieldnum) {
    // Record in solved file, or send to solved server.
    if (bp->solved_out) {
        logmsg("Field %i solved: writing to file %s to indicate this.\n", fieldnum, bp->solved_out);
        if (solvedfile_set(bp->solved_out, fieldnum)) {
            logerr("Failed to write solvedfile %s.\n", bp->solved_out);
        }
    }
    if (bp->solvedserver) {
        solvedclient_set(bp->fieldid, fieldnum);
    }
    // If we're just solving a single field, and we solved it...
    if (il_size(bp->fieldlist) == 1)
        bp->single_field_solved = TRUE;
}

static void free_matchobj(MatchObj* mo) {
    if (!mo) return;
    if (mo->sip) {
        sip_free(mo->sip);
        mo->sip = NULL;
    }
    if (mo->corr_field) {
        il_free(mo->corr_field);
    }
    if (mo->corr_index) {
        il_free(mo->corr_index);
    }
    if (mo->corr_index_rd) {
        dl_free(mo->corr_index_rd);
    }
    if (mo->corr_field_xy) {
        dl_free(mo->corr_field_xy);
    }
    if (mo->indexrdls) {
        free(mo->indexrdls);
        mo->indexrdls = NULL;
        mo->nindexrdls = 0;
    }
}

static void remove_duplicate_solutions(blind_t* bp) {
    int i, j;
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
            free_matchobj(mo2);
            bl_remove_index(bp->solutions, j);
        }
    }
}

static int write_solutions(blind_t* bp) {
    int i;

	if (bp->matchfname) {
		bp->mf = matchfile_open_for_writing(bp->matchfname);
		if (!bp->mf) {
			logerr("Failed to open file %s to write match file.\n", bp->matchfname);
            return -1;
		}
		boilerplate_add_fits_headers(bp->mf->header);
		qfits_header_add(bp->mf->header, "HISTORY", "This file was created by the program \"blind\".", NULL, NULL);
		qfits_header_add(bp->mf->header, "DATE", qfits_get_datetime_iso8601(), "Date this file was created.", NULL);
		add_blind_params(bp, bp->mf->header);
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
	}

	if (bp->indexrdlsfname) {
        qfits_header* h;
		bp->indexrdls = rdlist_open_for_writing(bp->indexrdlsfname);
		if (!bp->indexrdls) {
			logerr("Failed to open index RDLS file %s for writing.\n",
				   bp->indexrdlsfname);
            return -1;
		}
        h = rdlist_get_primary_header(bp->indexrdls);

        boilerplate_add_fits_headers(h);
        fits_add_long_history(h, "This \"indexrdls\" file was created by the program \"blind\"."
                              "  It contains the RA/DEC of index objects that were found inside a solved field.");
        qfits_header_add(h, "DATE", qfits_get_datetime_iso8601(), "Date this file was created.", NULL);
        add_blind_params(bp, h);
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
            if (rdlist_write_header(bp->indexrdls)) {
                logerr("Failed to write index RDLS field header.\n");
                return -1;
            }
            assert(mo->indexrdls);

            // HACK - should instead make mo.indexrdls an rd_t.
            rd_from_array(&rd, mo->indexrdls, mo->nindexrdls);

            if (rdlist_write_field(bp->indexrdls, &rd)) {
                logerr("Failed to write index RDLS entry.\n");
                return -1;
            }

            rd_free_data(&rd);

            if (rdlist_fix_header(bp->indexrdls)) {
                logerr("Failed to fix index RDLS field header.\n");
                return -1;
            }
        }

		if (rdlist_fix_primary_header(bp->indexrdls) ||
			rdlist_close(bp->indexrdls)) {
			logerr("Failed to close index RDLS file.\n");
            return -1;
		}
		bp->indexrdls = NULL;
	}

    if (bp->wcs_template) {
        // We want to write only the best WCS for each field.
        remove_duplicate_solutions(bp);

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

            boilerplate_add_fits_headers(hdr);
            qfits_header_add(hdr, "HISTORY", "This WCS header was created by the program \"blind\".", NULL, NULL);
            tm = qfits_get_datetime_iso8601();
            qfits_header_add(hdr, "DATE", tm, "Date this file was created.", NULL);
            add_blind_params(bp, hdr);
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
            fits_add_long_comment(hdr, "noverlap: %i", mo->noverlap);
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
    }

    // Note that this follows the WCS output, so we've eliminated all but the best solution.
    if (bp->corr_fname) {
        qfits_header* hdr;
        FILE* fid = fopen(bp->corr_fname, "wb");
        if (!fid) {
            logerr("Failed to open file %s to write correspondences.\n", bp->corr_fname);
            return -1;
        }

        hdr = qfits_table_prim_header_default();
        // FIXME boilerplate
        if (qfits_header_dump(hdr, fid)) {
            logerr("Failed to write FITS header to correspondence file %s\n", bp->corr_fname);
            return -1;
        }
        qfits_header_destroy(hdr);

        for (i=0; i<bl_size(bp->solutions); i++) {
            MatchObj* mo;
            int j;
            qfits_table* table;
            int datasize;
            int ncols, nrows, tablesize;
            int NC;
            int col;
            int corrs;

            mo = bl_access(bp->solutions, i);

            if (!(mo->corr_field_xy && mo->corr_index_rd)) {
                logerr("Match has no list of correspondences.\n");
                continue;
            }

            corrs = dl_size(mo->corr_field_xy) / 2;

            // field ra, dec, x, y
            // index ra, dec, x, y
            NC = 8;

            datasize = sizeof(double);
            ncols = NC;
            nrows = corrs;
            tablesize = datasize * nrows * ncols;
            table = qfits_table_new("", QFITS_BINTABLE, tablesize, ncols, nrows);
            table->tab_w = 0;

            col = 0;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "degrees", "field_ra");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "degrees", "field_dec");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "pixels", "field_x");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "pixels", "field_y");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "degrees", "index_ra");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "degrees", "index_dec");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "pixels", "index_x");
            col++;
            fits_add_column(table, col, TFITS_BIN_TYPE_D, 1, "pixels", "index_y");
            col++;

            assert(col == NC);

            table->tab_w = qfits_compute_table_width(table);

            hdr = qfits_table_ext_header_default(table);
            if (qfits_header_dump(hdr, fid)) {
                logerr("Failed to write FITS table header to correspondence file %s\n", bp->corr_fname);
                return -1;
            }
            qfits_header_destroy(hdr);
            qfits_table_close(table);

            //printf("Have %i field xy and %i index radecs.\n", dl_size(mo->corr_field_xy)/2, dl_size(mo->corr_index_rd)/2);
             
            for (j=0; j<corrs; j++) {
                double fxy[2];
                double fradec[2];
                double iradec[2];
                double ixy[2];
                bool ok;

                fxy[0] = dl_get(mo->corr_field_xy, j*2 + 0);
                fxy[1] = dl_get(mo->corr_field_xy, j*2 + 1);

                iradec[0] = dl_get(mo->corr_index_rd, j*2 + 0);
                iradec[1] = dl_get(mo->corr_index_rd, j*2 + 1);
                
                if (mo->sip)
                    sip_pixelxy2radec(mo->sip, fxy[0], fxy[1], fradec+0, fradec+1);
                else
                    tan_pixelxy2radec(&(mo->wcstan), fxy[0], fxy[1], fradec+0, fradec+1);

                if (mo->sip)
                    ok = sip_radec2pixelxy(mo->sip, iradec[0], iradec[1], ixy+0, ixy+1);
                else
                    ok = tan_radec2pixelxy(&(mo->wcstan), iradec[0], iradec[1], ixy+0, ixy+1);
                assert(ok);

                if (fits_write_data_D(fid, fradec[0]) ||
                    fits_write_data_D(fid, fradec[1]) ||
                    fits_write_data_D(fid, fxy[0]) ||
                    fits_write_data_D(fid, fxy[1]) ||
                    fits_write_data_D(fid, iradec[0]) ||
                    fits_write_data_D(fid, iradec[1]) ||
                    fits_write_data_D(fid, ixy[0]) ||
                    fits_write_data_D(fid, ixy[1])) {
                    logerr("Failed to write row %i to correspondence table %s.\n", j, bp->corr_fname);
                    return -1;
                }
                /*
                 printf("  pixels: (%g, %g) -- (%g, %g)\n",
                 fxy[0], fxy[1], ixy[0], ixy[1]);
                 printf("  RA/Dec: (%g, %g) -- (%g, %g)\n",
                 fradec[0], fradec[1], iradec[0], iradec[1]);
                 */
            }

            if (fits_pad_file(fid) ||
                fclose(fid)) {
                logerr("Failed to pad and close correspondence table %s: %s\n", bp->corr_fname, strerror(errno));
                return -1;
            }
            break;
        }
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

