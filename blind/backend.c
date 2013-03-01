/*
 This file is part of the Astrometry.net suite.
 Copyright 2007, 2008 Dustin Lang, Keir Mierle and Sam Roweis.
 Copyright 2010, 2011 Dustin Lang.

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
 * Accepts an augmented xylist that describes a field or set of fields to solve.
 * Reads a config file to find local indices, and merges information about the
 * indices with the job description to create an input file for 'blind'.  Runs blind
 * and merges the results.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <libgen.h>
#include <getopt.h>
#include <dirent.h>
#include <assert.h>

#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "blindutils.h"
#include "os-features.h"
#include "blind.h"
#include "log.h"
#include "qfits.h"
#include "errors.h"
#include "backend.h"
#include "tic.h"
#include "healpix.h"
#include "sip-utils.h"
#include "multiindex.h"

void backend_add_search_path(backend_t* backend, char* path) {
    sl_append(backend->index_paths, path);
}

char* backend_find_index(backend_t* backend, char* name) {
    int j;

    for (j=-1; j<sl_size(backend->index_paths); j++) {
        char* path;
        if (j == -1)
            if (strlen(name) && name[0] == '/') {
                // try as an absolute filename.
                path = strdup(name);
            } else {
                continue;
            }
        else
            asprintf_safe(&path, "%s/%s", sl_get(backend->index_paths, j), name);
        
        logverb("Trying path %s...\n", path);
        if (index_is_file_index(path))
            return path;
        free(path);
    }
    return NULL;
}

int backend_autoindex_search_paths(backend_t* backend) {
    int i;
    // Search the paths specified and add any indexes that are found.
    for (i=0; i<sl_size(backend->index_paths); i++) {
        char* path = sl_get(backend->index_paths, i);
        DIR* dir = opendir(path);
        sl* tryinds;
        int j;
        if (!dir) {
            logerr("Warning: failed to open index directory: \"%s\"\n", path);
            continue;
        }
        logverb("Auto-indexing directory \"%s\" ...\n", path);
        tryinds = sl_new(16);
        while (1) {
            struct dirent* de;
            char* name;
            char* fullpath;
            char* err;
            anbool ok;
            errno = 0;
            de = readdir(dir);
            if (!de) {
                if (errno)
                    SYSERROR("Failed to read entry from directory \"%s\"", path);
                break;
            }
            name = de->d_name;
            asprintf_safe(&fullpath, "%s/%s", path, name);
            if (path_is_dir(fullpath)) {
                logverb("Skipping directory %s\n", fullpath);
                free(fullpath);
                continue;
            }

            logverb("Checking file \"%s\"\n", fullpath);
            errors_start_logging_to_string();
            ok = index_is_file_index(fullpath);
            err = errors_stop_logging_to_string(": ");
            if (!ok) {
                logverb("File is not an index: %s\n", err);
                free(err);
                free(fullpath);
                continue;
            }
            free(err);

            sl_insert_sorted_nocopy(tryinds, fullpath);
        }
        closedir(dir);

        // add them in reverse order... (why?)
        for (j=sl_size(tryinds)-1; j>=0; j--) {
            char* path = sl_get(tryinds, j);
            logverb("Trying to add index \"%s\".\n", path);
            if (backend_add_index(backend, path))
                logmsg("Failed to add index \"%s\".\n", path);
        }
        sl_free2(tryinds);
    }
    return 0;
}

static int add_index(backend_t* backend, index_t* ind) {
	int k;
    // check that an index with the same id and healpix isn't already listed.
    for (k=0; k<pl_size(backend->indexes); k++) {
		index_t* m = pl_get(backend->indexes, k);
        if (m->indexid == ind->indexid &&
            m->healpix == ind->healpix) {
            logmsg("Warning: encountered two index files with the same INDEXID = %i and HEALPIX = %i: \"%s\" and \"%s\".  Keeping both.\n",
				   m->indexid, m->healpix, m->indexname, ind->indexname);
			//index_free(ind);
            //return 0;
        }
    }

	pl_append(backend->indexes, ind);

    // <= smallest we've seen?
	if (ind->index_scale_lower < backend->sizesmallest) {
		backend->sizesmallest = ind->index_scale_lower;
        bl_remove_all(backend->ismallest);
		il_append(backend->ismallest, pl_size(backend->indexes) - 1);
	} else if (ind->index_scale_lower == backend->sizesmallest) {
		il_append(backend->ismallest, pl_size(backend->indexes) - 1);
    }

    // >= largest we've seen?
	if (ind->index_scale_upper > backend->sizebiggest) {
		backend->sizebiggest = ind->index_scale_upper;
        bl_remove_all(backend->ibiggest);
		il_append(backend->ibiggest, pl_size(backend->indexes) - 1);
	} else if (ind->index_scale_upper == backend->sizebiggest) {
		il_append(backend->ibiggest, pl_size(backend->indexes) - 1);
	}
	return 0;
}

int backend_add_index(backend_t* backend, char* path) {
    int k;
    index_t* ind = NULL;
    char* quadpath = index_get_quad_filename(path);
    char* base = basename_safe(quadpath);
	double t0;
    free(quadpath);

    // check that an index with the same filename hasn't already been added.
    for (k=0; k<pl_size(backend->indexes); k++) {
		ind = pl_get(backend->indexes, k);
        // ind->indexname is a path to the quad filename; strip off directory component.
        char* mbase = basename_safe(ind->indexname);
        anbool eq = streq(base, mbase);
        free(mbase);
        if (eq) {
            logmsg("Warning: we've already seen an index with the same name: \"%s\".  Adding it anyway...\n", ind->indexname);
            //free(base);
            //return 0;
        }
    }
    free(base);

	t0 = timenow();
	ind = index_load(path, backend->inparallel ? 0 : INDEX_ONLY_LOAD_METADATA, NULL);
	debug("index_load(\"%s\") took %g ms\n", path, 1000 * (timenow() - t0));
	if (!ind) {
		ERROR("Failed to load index from path %s", path);
		return -1;
	}
	if (add_index(backend, ind)) {
		ERROR("Failed to add index \"%s\"", path);
		return -1;
	}
	pl_append(backend->free_indexes, ind);
    return 0;
}

static void add_index_to_blind(backend_t* backend, blind_t* bp,
                               int i) {
	index_t* index;
	index = pl_get(backend->indexes, i);
    if (backend->inparallel) {
        blind_add_loaded_index(bp, index);
    } else {
        blind_add_index(bp, index->indexname);
    }
}

int backend_parse_config_file(backend_t* backend, const char* fn) {
	FILE* fconf;
    int rtn;
	fconf = fopen(fn, "r");
	if (!fconf) {
		SYSERROR("Failed to open config file \"%s\"", fn);
        return -1;
	}
    rtn = backend_parse_config_file_stream(backend, fconf);
	fclose(fconf);
    return rtn;
}

int backend_parse_config_file_stream(backend_t* backend, FILE* fconf) {
    sl* indices = sl_new(16);
    sl* mindices = sl_new(16);
    anbool auto_index = FALSE;
    int i;
    int rtn = 0;

	while (1) {
		char buffer[10240];
		char* nextword;
		char* line;
		if (!fgets(buffer, sizeof(buffer), fconf)) {
			if (feof(fconf))
				break;
			SYSERROR("Failed to read a line from the config file");
            rtn = -1;
            goto done;
		}
		line = buffer;
		// strip off newline
		if (line[strlen(line) - 1] == '\n')
			line[strlen(line) - 1] = '\0';
		// skip leading whitespace:
		while (*line && isspace(*line))
			line++;
		// skip comments
		if (line[0] == '#')
			continue;
		// skip blank lines.
		if (line[0] == '\0')
			continue;

		if (is_word(line, "index ", &nextword)) {
            // don't try to find the index yet - because search paths may be
            // added later.
            sl_append(indices, nextword);
		} else if (is_word(line, "multiindex ", &nextword)) {
            // don't try to find the index yet - because search paths may be
            // added later.
			sl_append(mindices, nextword);
        } else if (is_word(line, "autoindex", &nextword)) {
            auto_index = TRUE;
		} else if (is_word(line, "inparallel", &nextword)) {
			backend->inparallel = TRUE;
		} else if (is_word(line, "minwidth ", &nextword)) {
			backend->minwidth = atof(nextword);
		} else if (is_word(line, "maxwidth ", &nextword)) {
			backend->maxwidth = atof(nextword);
		} else if (is_word(line, "cpulimit ", &nextword)) {
			backend->cpulimit = atof(nextword);
		} else if (is_word(line, "depths ", &nextword)) {
            if (parse_depth_string(backend->default_depths, nextword)) {
                rtn = -1;
                goto done;
            }
		} else if (is_word(line, "add_path ", &nextword)) {
            backend_add_search_path(backend, nextword);
		} else {
			ERROR("Didn't understand this config file line: \"%s\"", line);
			// unknown config line is a firing offense
            rtn = -1;
            goto done;
		}
	}

    for (i=0; i<sl_size(indices); i++) {
        char* ind = sl_get(indices, i);
        char* path;
        logverb("Trying index %s...\n", ind);

        path = backend_find_index(backend, ind);
        if (!path) {
            logmsg("Couldn't find index \"%s\".\n", ind);
            rtn = -1;
            goto done;
        }
        if (backend_add_index(backend, path))
            logmsg("Failed to add index \"%s\".\n", path);
		free(path);
    }

    for (i=0; i<sl_size(mindices); i++) {
        char* ind = sl_get(mindices, i);
        char* path;
		char* skdt;
		char* skdtpath;
		int j;
		sl* words = sl_split(NULL, ind, " ");
		multiindex_t* mi;

		if (sl_size(words) < 2) {
			logmsg("Config line 'multiindex' must be followed by skdt and inds\n");
            rtn = -1;
            goto done;
		}
		skdt = sl_get(words, 0);
		sl_remove(words, 0);
		{
			char* s = sl_join(words, " / ");
			logverb("Trying multi-index %s + %s...\n", skdt, s);
			free(s);
		}
		skdtpath = backend_find_index(backend, skdt);
        if (!skdtpath) {
            logmsg("Couldn't find skdt \"%s\".\n", skdt);
            rtn = -1;
            goto done;
        }
		for (j=0; j<sl_size(words); j++) {
			ind = sl_get(words, j);
			path = backend_find_index(backend, ind);
			if (!path) {
				logmsg("Couldn't find index \"%s\".\n", path);
				rtn = -1;
				goto done;
			}
			sl_set(words, j, path);
            // sl_set makes a copy.
            free(path);
		}

		mi = multiindex_open(skdtpath, words);
		if (!mi) {
			char* s = sl_join(words, " / ");
			logerr("Failed to open multiindex: %s + %s\n", skdt, s);
			free(s);
			rtn = -1;
			goto done;
		}
		for (j=0; j<multiindex_n(mi); j++) {
			index_t* ind = multiindex_get(mi, j);
			if (add_index(backend, ind)) {
				ERROR("Failed to add index \"%s\"", sl_get(words, j));
				return -1;
			}
		}
		pl_append(backend->free_mindexes, mi);
        sl_free2(words);
        free(skdt);
        free(skdtpath);
    }

    if (auto_index) {
        backend_autoindex_search_paths(backend);
    }

 done:
    sl_free2(indices);
    sl_free2(mindices);
	return rtn;
}

static job_t* job_new() {
	job_t* job = calloc(1, sizeof(job_t));
	if (!job) {
		SYSERROR("Failed to allocate a new job_t.");
		return NULL;
	}
	job->scales = dl_new(8);
	job->depths = il_new(8);
	return job;
}

void job_free(job_t* job) {
	if (!job)
		return;
	dl_free(job->scales);
	il_free(job->depths);
	free(job);
}

static double job_imagew(job_t* job) {
    return job->bp.solver.field_maxx;
}
static double job_imageh(job_t* job) {
    return job->bp.solver.field_maxy;
}

int backend_run_job(backend_t* backend, job_t* job) {
    blind_t* bp = &(job->bp);
    solver_t* sp = &(bp->solver);
    
    int i;
    double app_min_default;
    double app_max_default;
    anbool solved = FALSE;

    if (blind_is_run_obsolete(bp, sp)) {
        goto finish;
    }

    app_min_default = deg2arcsec(backend->minwidth) / job_imagew(job);
    app_max_default = deg2arcsec(backend->maxwidth) / job_imagew(job);

    if (backend->inparallel)
        bp->indexes_inparallel = TRUE;

	if (job->use_radec_center) {
		logmsg("Only searching for solutions within %g degrees of RA,Dec (%g,%g)\n",
			   job->search_radius, job->ra_center, job->dec_center);
		solver_set_radec(sp, job->ra_center, job->dec_center, job->search_radius);
	}

    for (i=0; i<il_size(job->depths)/2; i++) {
		int startobj = il_get(job->depths, i*2);
        int endobj = il_get(job->depths, i*2+1);
        int j;

        if (startobj || endobj) {
            // make depth ranges be inclusive.
            endobj++;
            // up to this point they are 1-indexed, but with default value
            // zero; blind uses 0-indexed.
            if (startobj)
                startobj--;
            if (endobj)
                endobj--;
        }

		for (j=0; j<dl_size(job->scales) / 2; j++) {
			double fmin, fmax;
			double app_max, app_min;
            int k;
            il* indexlist;

			// arcsec per pixel range
			app_min = dl_get(job->scales, j * 2);
			app_max = dl_get(job->scales, j * 2 + 1);
            if (app_min == 0.0)
                app_min = app_min_default;
            if (app_max == 0.0)
                app_max = app_max_default;
            sp->funits_lower = app_min;
            sp->funits_upper = app_max;

            sp->startobj = startobj;
			if (endobj)
                sp->endobj = endobj;

			// minimum quad size to try (in pixels)
            sp->quadsize_min = bp->quad_size_fraction_lo *
                MIN(job_imagew(job), job_imageh(job));

			// range of quad sizes that could be found in the field,
			// in arcsec.
            // the hypotenuse...
			fmax = bp->quad_size_fraction_hi *
                hypot(job_imagew(job), job_imageh(job)) * app_max;
			fmin = sp->quadsize_min * app_min;

			// Select the indices that should be checked.
            indexlist = il_new(16);
			for (k = 0; k < pl_size(backend->indexes); k++) {
				index_t* index = pl_get(backend->indexes, k);
                if (!index_overlaps_scale_range(index, fmin, fmax))
                    continue;
                il_append(indexlist, k);
			}

			// Use the (list of) smallest or largest indices if no other one fits.
			if (!il_size(indexlist)) {
                il* list = NULL;
                if (fmin > backend->sizebiggest) {
                    list = backend->ibiggest;
                } else if (fmax < backend->sizesmallest) {
                    list = backend->ismallest;
                } else {
                    assert(0);
                }
                il_append_list(indexlist, list);
            }

            for (k=0; k<il_size(indexlist); k++) {
                int ii = il_get(indexlist, k);
				index_t* index = pl_get(backend->indexes, ii);
                anbool inrange = TRUE;
				if (job->use_radec_center)
					inrange = index_is_within_range(index, job->ra_center, job->dec_center, job->search_radius);
                if (!inrange) {
                    logverb("Not using index %s because it's not within %g degrees of (RA,Dec) = (%g,%g)\n",
                            index->indexname, job->search_radius, job->ra_center, job->dec_center);
					continue;
				}
				add_index_to_blind(backend, bp, ii);
            }

            il_free(indexlist);

            logverb("Running blind solver:\n");
            blind_log_run_parameters(bp);

            blind_run(bp);

            // we only want to try using the verify_wcses the first time.
            blind_clear_verify_wcses(bp);
            blind_clear_indexes(bp);
            blind_clear_solutions(bp);
            blind_clear_indexes(bp);
            solver_clear_indexes(sp);

            if (blind_is_run_obsolete(bp, sp)) {
                solved = TRUE;
                break;
            }
		}
        if (solved)
            break;
	}

	logverb("cx<=dx constraints: %i\n", sp->num_cxdx_skipped);
	logverb("meanx constraints: %i\n", sp->num_meanx_skipped);
	logverb("RA,Dec constraints: %i\n", sp->num_radec_skipped);
	logverb("AB scale constraints: %i\n", sp->num_abscale_skipped);

 finish:
    solver_cleanup(sp);
    blind_cleanup(bp);
	return 0;
}

static void parse_sip_coeffs(const qfits_header* hdr, const char* prefix, sip_t* wcs) {
	char key[64];
	int order, i, j;
	sprintf(key, "%sSAO", prefix);
	order = qfits_header_getint(hdr, key, -1);
	if (order >= 2) {
		if (order > 9)
			order = 9;
		wcs->a_order = order;
		wcs->b_order = order;
		for (i=0; i<=order; i++) {
			for (j=0; (i+j)<=order; j++) {
				if (i+j < 1)
					continue;
				sprintf(key, "%sA%i%i", prefix, i, j);
				wcs->a[i][j] = qfits_header_getdouble(hdr, key, 0.0);
				sprintf(key, "%sB%i%i", prefix, i, j);
				wcs->b[i][j] = qfits_header_getdouble(hdr, key, 0.0);
			}
		}
	}
	sprintf(key, "%sSAPO", prefix);
	order = qfits_header_getint(hdr, key, -1);
	if (order >= 2) {
		if (order > 9)
			order = 9;
		wcs->ap_order = order;
		wcs->bp_order = order;
		for (i=0; i<=order; i++) {
			for (j=0; (i+j)<=order; j++) {
				if (i+j < 1)
					continue;
				sprintf(key, "%sAP%i%i", prefix, i, j);
				wcs->ap[i][j] = qfits_header_getdouble(hdr, key, 0.0);
				sprintf(key, "%sBP%i%i", prefix, i, j);
				wcs->bp[i][j] = qfits_header_getdouble(hdr, key, 0.0);
			}
		}
	}
}

static anbool parse_job_from_qfits_header(const qfits_header* hdr, job_t* job) {
    blind_t* bp = &(job->bp);
    solver_t* sp = &(bp->solver);

	double dnil = -HUGE_VAL;
	char *pstr;
	int n;
    anbool run;

    anbool default_tweak = TRUE;
    int default_tweakorder = 2;
    double default_odds_toprint = 1e6;
    double default_odds_tokeep = 1e9;
    double default_odds_tosolve = 1e9;
    double default_odds_totune = 1e6;
    //double default_image_fraction = 1.0;
    char* fn;
    double val;

    blind_init(bp);
    // must be in this order because init_parameters handily zeros out sp
    solver_set_default_values(sp);

	// Here we assume that the field's pixel coordinataes go from zero to IMAGEW,H.
    sp->field_maxx = qfits_header_getdouble(hdr, "IMAGEW", dnil);
    sp->field_maxy = qfits_header_getdouble(hdr, "IMAGEH", dnil);
	if ((sp->field_maxx == dnil) || (sp->field_maxy == dnil) ||
		(sp->field_maxx <= 0.0) || (sp->field_maxy <= 0.0)) {
		logerr("Must specify positive \"IMAGEW\" and \"IMAGEH\".\n");
		goto bailout;
	}

	sp->verify_uniformize = qfits_header_getboolean(hdr, "ANVERUNI", sp->verify_uniformize);
	sp->verify_dedup = qfits_header_getboolean(hdr, "ANVERDUP", sp->verify_dedup);

    val = qfits_header_getdouble(hdr, "ANPOSERR", 0.0);
    if (val > 0.0)
        sp->verify_pix = val;
    val = qfits_header_getdouble(hdr, "ANCTOL", 0.0);
    if (val > 0.0)
        sp->codetol = val;
    val = qfits_header_getdouble(hdr, "ANDISTR", 0.0);
    if (val > 0.0)
        sp->distractor_ratio = val;

    blind_set_solvedout_file  (bp, fn=fits_get_long_string(hdr, "ANSOLVED"));
    free(fn);
    blind_set_solvedin_file  (bp, fn=fits_get_long_string(hdr, "ANSOLVIN"));
    free(fn);
    blind_set_match_file   (bp, fn=fits_get_long_string(hdr, "ANMATCH" ));
    free(fn);
    blind_set_rdls_file    (bp, fn=fits_get_long_string(hdr, "ANRDLS"  ));
    free(fn);
    blind_set_scamp_file   (bp, fn=fits_get_long_string(hdr, "ANSCAMP" ));
    free(fn);
    blind_set_wcs_file     (bp, fn=fits_get_long_string(hdr, "ANWCS"   ));
    free(fn);
    blind_set_corr_file    (bp, fn=fits_get_long_string(hdr, "ANCORR"  ));
    free(fn);
    blind_set_cancel_file  (bp, fn=fits_get_long_string(hdr, "ANCANCEL"));
    free(fn);

    blind_set_xcol(bp, fn=fits_get_dupstring(hdr, "ANXCOL"));
    free(fn);
    blind_set_ycol(bp, fn=fits_get_dupstring(hdr, "ANYCOL"));
    free(fn);

    bp->timelimit = qfits_header_getint(hdr, "ANTLIM", 0);
    bp->cpulimit = qfits_header_getdouble(hdr, "ANCLIM", 0.0);
    bp->logratio_tosolve = log(qfits_header_getdouble(hdr, "ANODDSSL", default_odds_tosolve));
	logverb("Set odds ratio to solve to %g (log = %g)\n", exp(bp->logratio_tosolve), bp->logratio_tosolve);


    sp->logratio_toprint = log(qfits_header_getdouble(hdr, "ANODDSPR", default_odds_toprint));
    sp->logratio_tokeep = log(qfits_header_getdouble(hdr, "ANODDSKP", default_odds_tokeep));
    sp->logratio_totune = log(qfits_header_getdouble(hdr, "ANODDSTU", default_odds_totune));
    sp->logratio_bail_threshold = log(qfits_header_getdouble(hdr, "ANODDSBL", DEFAULT_BAIL_THRESHOLD));
	val = qfits_header_getdouble(hdr, "ANODDSST", 0.0);
	if (val > 0.0)
		sp->logratio_stoplooking = log(val);
    bp->best_hit_only = TRUE;

	// gotta keep it to solve it!
	sp->logratio_tokeep = MIN(sp->logratio_tokeep, bp->logratio_tosolve);
	// gotta print it to keep it (so what if that doesn't make sense)!
	sp->logratio_toprint = MIN(sp->logratio_toprint, sp->logratio_tokeep);

	// job->image_fraction = qfits_header_getdouble(hdr, "ANIMFRAC", job->image_fraction);
    job->include_default_scales = qfits_header_getboolean(hdr, "ANAPPDEF", 0);

    sp->parity = PARITY_BOTH;
	pstr = qfits_pretty_string(qfits_header_getstr(hdr, "ANPARITY"));
	if (pstr && streq(pstr, "NEG"))
		sp->parity = PARITY_FLIP;
	else if (pstr && streq(pstr, "POS"))
		sp->parity = PARITY_NORMAL;

	sp->set_crpix_center = qfits_header_getboolean(hdr, "ANCRPIXC", FALSE);
	sp->crpix[0] = qfits_header_getint(hdr, "ANCRPIX1", sp->crpix[0]);
	sp->crpix[1] = qfits_header_getint(hdr, "ANCRPIX2", sp->crpix[1]);
	sp->set_crpix = (sp->set_crpix_center || 
					 // were the values set?
					 qfits_header_getstr(hdr, "ANCRPIX1") ||
					 qfits_header_getstr(hdr, "ANCRPIX2"));

    if (qfits_header_getboolean(hdr, "ANTWEAK", default_tweak)) {
        int order = qfits_header_getint(hdr, "ANTWEAKO", default_tweakorder);
        //bp->do_tweak = TRUE;
        sp->do_tweak = TRUE;
        sp->tweak_aborder = order;
        sp->tweak_abporder = order;
    }

    val = qfits_header_getdouble(hdr, "ANQSFMIN", 0.0);
    if (val > 0.0)
        bp->quad_size_fraction_lo = val;
    val = qfits_header_getdouble(hdr, "ANQSFMAX", 0.0);
    if (val > 0.0)
        bp->quad_size_fraction_hi = val;

    job->ra_center = qfits_header_getdouble(hdr, "ANERA", HUGE_VAL);
    job->dec_center = qfits_header_getdouble(hdr, "ANEDEC", HUGE_VAL);
    job->search_radius = qfits_header_getdouble(hdr, "ANERAD", HUGE_VAL);
    job->use_radec_center = ((job->ra_center     != HUGE_VAL) &&
                             (job->dec_center    != HUGE_VAL) &&
                             (job->search_radius != HUGE_VAL));

	// tag-along columns
	bp->rdls_tagalong_all = qfits_header_getboolean(hdr, "ANTAGALL", FALSE);
	if (!bp->rdls_tagalong_all) {
		n = 1;
		while (1) {
			char key[64];
			char* val;
			sprintf(key, "ANTAG%i", n);
			val = fits_get_dupstring(hdr, key);
			if (!val)
				break;
			if (!bp->rdls_tagalong)
				bp->rdls_tagalong = sl_new(16);
			sl_append_nocopy(bp->rdls_tagalong, val);
			n++;
		}
	}

	// sort RDLS column
	bp->sort_rdls = fits_get_dupstring(hdr, "ANRDSORT");

	n = 1;
	while (1) {
		char key[64];
		double lo, hi;
		sprintf(key, "ANAPPL%i", n);
		lo = qfits_header_getdouble(hdr, key, dnil);
		sprintf(key, "ANAPPU%i", n);
		hi = qfits_header_getdouble(hdr, key, dnil);
		if ((hi == dnil) && (lo == dnil))
			break;
        if ((lo != dnil) && (hi != dnil)) {
            if ((lo < 0) || (lo > hi)) {
                logerr("Scale range %g to %g is invalid: min must be >= 0, max must be >= min.\n", lo, hi);
                goto bailout;
            }
        }
        if (hi == dnil)
            hi = 0.0;
        if (lo == dnil)
            lo = 0.0;
		dl_append(job->scales, lo);
		dl_append(job->scales, hi);
		n++;
	}

	n = 1;
	while (1) {
		char key[64];
		int dlo, dhi;
		sprintf(key, "ANDPL%i", n);
		dlo = qfits_header_getint(hdr, key, 0);
		sprintf(key, "ANDPU%i", n);
		dhi = qfits_header_getint(hdr, key, 0);
		if (dlo == 0 && dhi == 0)
			break;
        if ((dlo < 1) || (dlo > dhi)) {
            logerr("Depth range %i to %i is invalid: min must be >= 1, max must be >= min.\n", dlo, dhi);
            goto bailout;
        }
		il_append(job->depths, dlo);
		il_append(job->depths, dhi);
		n++;
	}

	n = 1;
	while (1) {
		char lokey[64];
		char hikey[64];
		int lo, hi;
		sprintf(lokey, "ANFDL%i", n);
		lo = qfits_header_getint(hdr, lokey, -1);
		if (lo == -1)
			break;
		sprintf(hikey, "ANFDU%i", n);
		hi = qfits_header_getint(hdr, hikey, -1);
		if (hi == -1)
			break;
        if ((lo <= 0) || (lo > hi)) {
            logerr("Field range %i to %i is invalid: min must be >= 1, max must be >= min.\n", lo, hi);
            logmsg("  (FITS headers: \"%s = %s\", \"%s = %s\")\n",
                   lokey, qfits_pretty_string(qfits_header_getstr(hdr, lokey)),
                   hikey, qfits_pretty_string(qfits_header_getstr(hdr, hikey)));
            goto bailout;
        }

        blind_add_field_range(bp, lo, hi);
		n++;
	}

	n = 1;
	while (1) {
		char key[64];
		int fld;
		sprintf(key, "ANFD%i", n);
		fld = qfits_header_getint(hdr, key, -1);
		if (fld == -1)
			break;
        if (fld <= 0) {
            logerr("Field %i is invalid: must be >= 1.  (FITS header: \"%s = %s\")\n", fld, key,
                   qfits_pretty_string(qfits_header_getstr(hdr, key)));
            goto bailout;
        }

        blind_add_field(bp, fld);
		n++;
	}

	n = 1;
	while (1) {
		char key[64];
        sip_t wcs;
		char* keys[] = { "ANW%iPIX1", "ANW%iPIX2", "ANW%iVAL1", "ANW%iVAL2",
				 "ANW%iCD11", "ANW%iCD12", "ANW%iCD21", "ANW%iCD22" };
		double* vals[] = { &(wcs.wcstan. crval[0]), &(wcs.wcstan.crval[1]),
				   &(wcs.wcstan.crpix[0]), &(wcs.wcstan.crpix[1]),
				   &(wcs.wcstan.cd[0][0]), &(wcs.wcstan.cd[0][1]),
				   &(wcs.wcstan.cd[1][0]), &(wcs.wcstan.cd[1][1]) };
		int j;
		int bail = 0;
        memset(&wcs, 0, sizeof(wcs));
		for (j = 0; j < 8; j++) {
			sprintf(key, keys[j], n);
			*(vals[j]) = qfits_header_getdouble(hdr, key, dnil);
			if (*(vals[j]) == dnil) {
				bail = 1;
				break;
			}
		}
		if (bail)
			break;

        // SIP terms
		sprintf(key, "ANW%i", n);
		parse_sip_coeffs(hdr, key, &wcs);

		sip_ensure_inverse_polynomials(&wcs);

        blind_add_verify_wcs(bp, &wcs);
		n++;
	}

	// Distortion to apply before matching...
	do {
		sip_t dsip;
		double p0, p1;
		memset(&dsip, 0, sizeof(sip_t));
		p0 = qfits_header_getdouble(hdr, "ANDPIX0", dnil);
		if (p0 == dnil)
			break;
		p1 = qfits_header_getdouble(hdr, "ANDPIX1", dnil);
		if (p1 == dnil)
			break;
		dsip.wcstan.crpix[0] = p0;
		dsip.wcstan.crpix[1] = p1;
		parse_sip_coeffs(hdr, "AND", &dsip);
		if ((dsip.a_order > 1 && dsip.b_order > 1) ||
			(dsip.ap_order > 1 && dsip.bp_order > 1)) {
			sp->predistort = malloc(sizeof(sip_t));
			memcpy(sp->predistort, &dsip, sizeof(sip_t));
		}
	} while (0);

	run = qfits_header_getboolean(hdr, "ANRUN", FALSE);

	// Default: solve first field.
	if (run && !il_size(bp->fieldlist)) {
        blind_add_field(bp, 1);
	}

    return TRUE;

 bailout:
    return FALSE;
}



backend_t* backend_new() {
	backend_t* backend = calloc(1, sizeof(backend_t));
	backend->index_paths = sl_new(10);
    backend->indexes = pl_new(16);
    backend->free_indexes = pl_new(16);
    backend->free_mindexes = pl_new(16);
	backend->ismallest = il_new(4);
	backend->ibiggest = il_new(4);
	backend->default_depths = il_new(4);
	backend->sizesmallest = HUGE_VAL;
	backend->sizebiggest = -HUGE_VAL;

	// Default scale estimate: field width, in degrees:
	backend->minwidth = 0.1;
	backend->maxwidth = 180.0;
    backend->cpulimit = 600.0;
	return backend;
}

void backend_free(backend_t* backend) {
	int i;
    if (!backend)
        return;
    if (backend->free_indexes) {
        for (i=0; i<pl_size(backend->free_indexes); i++) {
            index_t* ind = pl_get(backend->free_indexes, i);
            index_free(ind);
        }
        pl_free(backend->free_indexes);
    }
    if (backend->free_mindexes) {
        for (i=0; i<pl_size(backend->free_mindexes); i++) {
            multiindex_t* mi = pl_get(backend->free_mindexes, i);
            multiindex_free(mi);
        }
        pl_free(backend->free_mindexes);
    }
	pl_free(backend->indexes);
    if (backend->ismallest)
        il_free(backend->ismallest);
    if (backend->ibiggest)
        il_free(backend->ibiggest);
    if (backend->default_depths)
        il_free(backend->default_depths);
    if (backend->index_paths)
        sl_free2(backend->index_paths);
    free(backend);
}

job_t* backend_read_job_file(backend_t* backend, const char* jobfn) {
    qfits_header* hdr;
    job_t* job;
    blind_t* bp;

    // Read primary header.
    hdr = qfits_header_read(jobfn);
    if (!hdr) {
        ERROR("Failed to parse FITS header from file \"%s\"", jobfn);
        return NULL;
    }
    job = job_new();
    if (!parse_job_from_qfits_header(hdr, job)) {
        job_free(job);
        qfits_header_destroy(hdr);
        return NULL;
    }
    qfits_header_destroy(hdr);

    bp = &(job->bp);

    blind_set_field_file(bp, jobfn);

    // If the job has no scale estimate, search everything provided
    // by the backend
    if (!dl_size(job->scales) || job->include_default_scales) {
        double arcsecperpix;
        arcsecperpix = deg2arcsec(backend->minwidth) / job_imagew(job);
        dl_append(job->scales, arcsecperpix);
        arcsecperpix = deg2arcsec(backend->maxwidth) / job_imagew(job);
        dl_append(job->scales, arcsecperpix);
    }

    // The job can only decrease the CPU limit.
    if ((bp->cpulimit == 0.0) || bp->cpulimit > backend->cpulimit) {
        logverb("Decreasing CPU time limit to the backend's limit of %g seconds\n",
                backend->cpulimit);
        bp->cpulimit = backend->cpulimit;
    }

    // If the job didn't specify depths, set defaults.
    if (il_size(job->depths) == 0) {
        if (backend->inparallel) {
            // no limit.
            il_append(job->depths, 0);
            il_append(job->depths, 0);
        } else
            il_append_list(job->depths, backend->default_depths);
    }

    if (backend->cancelfn)
        blind_set_cancel_file(bp, backend->cancelfn);
    if (backend->solvedfn)
        blind_set_solved_file(bp, backend->solvedfn);

    return job;
}

void job_set_cancel_file(job_t* job, const char* fn) {
    blind_set_cancel_file(&(job->bp), fn);
}

void job_set_solved_file(job_t* job, const char* fn) {
    blind_set_solved_file(&(job->bp), fn);
}

// Modify all filenames to be relative to "dir".
int job_set_base_dir(job_t* job, const char* dir) {
	return job_set_output_base_dir(job, dir) ||
		job_set_input_base_dir(job, dir);
}

int job_set_input_base_dir(job_t* job, const char* dir) {
    char* path;
    blind_t* bp = &(job->bp);
    logverb("Changing input file base dir to %s\n", dir);
    if (bp->fieldfname) {
        path = resolve_path(bp->fieldfname, dir);
        logverb("Changing %s to %s\n", bp->fieldfname, path);
        blind_set_field_file(bp, path);
    }
	return 0;
}

int job_set_output_base_dir(job_t* job, const char* dir) {
    char* path;
    blind_t* bp = &(job->bp);
    logverb("Changing output file base dir to %s\n", dir);
    if (bp->cancelfname) {
        path = resolve_path(bp->cancelfname, dir);
        logverb("Cancel file was %s, changing to %s.\n", bp->cancelfname, path);
        blind_set_cancel_file(bp, path);
    }
    if (bp->solved_in) {
        path = resolve_path(bp->solved_in, dir);
        logverb("Changing %s to %s\n", bp->solved_in, path);
        blind_set_solvedin_file(bp, path);
    }
    if (bp->solved_out) {
        path = resolve_path(bp->solved_out, dir);
        logverb("Changing %s to %s\n", bp->solved_out, path);
        blind_set_solvedout_file(bp, path);
    }
    if (bp->matchfname) {
        path = resolve_path(bp->matchfname, dir);
        logverb("Changing %s to %s\n", bp->matchfname, path);
        blind_set_match_file(bp, path);
    }
    if (bp->indexrdlsfname) {
        path = resolve_path(bp->indexrdlsfname, dir);
        logverb("Changing %s to %s\n", bp->indexrdlsfname, path);
        blind_set_rdls_file(bp, path);
    }
    if (bp->scamp_fname) {
        path = resolve_path(bp->scamp_fname, dir);
        logverb("Changing %s to %s\n", bp->scamp_fname, path);
        blind_set_scamp_file(bp, path);
    }
    if (bp->corr_fname) {
        path = resolve_path(bp->corr_fname, dir);
        logverb("Changing %s to %s\n", bp->corr_fname, path);
        blind_set_corr_file(bp, path);
    }
    if (bp->wcs_template) {
        path = resolve_path(bp->wcs_template, dir);
        logverb("Changing %s to %s\n", bp->wcs_template, path);
        blind_set_wcs_file(bp, path);
    }
    return 0;
}

