/*
 This file is part of the Astrometry.net suite.
 Copyright 2007, 2008 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include "fileutil.h"
#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "scriptutils.h"
#include "gnu-specific.h"
#include "blind.h"
#include "log.h"
#include "qfits.h"
#include "errors.h"

static bool verbose = FALSE;

static struct option long_options[] =
    {
	    {"help",    no_argument,       0, 'h'},
        {"verbose", no_argument,       0, 'v'},
	    {"config",  required_argument, 0, 'c'},
	    {"cancel",  required_argument, 0, 'C'},
        {"to-stderr", no_argument,     0, 'E'},
	    {0, 0, 0, 0}
    };

static const char* OPTIONS = "hc:i:vC:E";

static void print_help(const char* progname) {
	printf("Usage:   %s [options] <augmented xylist>\n"
	       "   [-c <backend config file>]  (default: \"backend.cfg\" in the directory ../etc/ relative to the directory containing the \"backend\" executable)\n"
           "   [-C <cancel-filename>]: quit solving if the file <cancel-filename> appears.\n"
           "   [-v]: verbose\n"
           "   [-E]: send log messages to stderr\n"
	       "\n", progname);
}

struct indexinfo {
	char* indexname;
    int indexid;
    int healpix;
    int hpnside;
	// quad size, in arcsec.
	double losize;
	double hisize;
};
typedef struct indexinfo indexinfo_t;

struct backend {
	bl* indexinfos;
	sl* index_paths;
	il* ibiggest;
	il* ismallest;
    il* default_depths;
	double sizesmallest;
	double sizebiggest;
	bool inparallel;
	double minwidth;
	double maxwidth;
    int cpulimit;
};
typedef struct backend backend_t;

static int add_index(backend_t* backend, char* path) {
	indexinfo_t ii;
    int k;

    if (index_get_scale_and_id(path, &ii.losize, &ii.hisize,
                               &ii.indexid, &ii.healpix, &ii.hpnside)) {
        ERROR("Failed to get index metadata for index %s", path);
        return -1;
    }

    // check that an index with the same id and healpix isn't already listed.
    for (k=0; k<bl_size(backend->indexinfos); k++) {
        indexinfo_t* iii = bl_access(backend->indexinfos, k);
        if (iii->indexid == ii.indexid &&
            iii->healpix == ii.healpix) {
            logverb("Skipping duplicate index %s\n", path);
            return 0;
        }
    }

	ii.indexname = strdup(path);

	bl_append(backend->indexinfos, &ii);

    // <= smallest we've seen?
	if (ii.losize < backend->sizesmallest) {
		backend->sizesmallest = ii.losize;
        bl_remove_all(backend->ismallest);
		il_append(backend->ismallest, bl_size(backend->indexinfos) - 1);
	} else if (ii.losize == backend->sizesmallest) {
		il_append(backend->ismallest, bl_size(backend->indexinfos) - 1);
    }

    // >= largest we've seen?
	if (ii.hisize > backend->sizebiggest) {
		backend->sizebiggest = ii.hisize;
        bl_remove_all(backend->ibiggest);
		il_append(backend->ibiggest, bl_size(backend->indexinfos) - 1);
	} else if (ii.hisize == backend->sizebiggest) {
		il_append(backend->ibiggest, bl_size(backend->indexinfos) - 1);
	}
    return 0;
}

static int parse_config_file(FILE* fconf, backend_t* backend) {
    sl* indices = sl_new(16);
    bool auto_index = FALSE;
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
        } else if (is_word(line, "autoindex", &nextword)) {
            auto_index = TRUE;
		} else if (is_word(line, "inparallel", &nextword)) {
			backend->inparallel = TRUE;
		} else if (is_word(line, "minwidth ", &nextword)) {
			backend->minwidth = atof(nextword);
		} else if (is_word(line, "maxwidth ", &nextword)) {
			backend->maxwidth = atof(nextword);
		} else if (is_word(line, "cpulimit ", &nextword)) {
			backend->cpulimit = atoi(nextword);
		} else if (is_word(line, "depths ", &nextword)) {
            if (parse_depth_string(backend->default_depths, nextword)) {
                rtn = -1;
                goto done;
            }
		} else if (is_word(line, "add_path ", &nextword)) {
			sl_append(backend->index_paths, nextword);
		} else {
			ERROR("Didn't understand this config file line: \"%s\"", line);
			// unknown config line is a firing offense
            rtn = -1;
            goto done;
		}
	}

    for (i=0; i<sl_size(indices); i++) {
        int j;
        char* ind = sl_get(indices, i);
        bool found = FALSE;
        logverb("Trying index %s...\n", ind);

        for (j=-1; j<sl_size(backend->index_paths); j++) {
            char* path;
            if (j == -1)
                if (strlen(ind) && ind[0] == '/') {
                    // try as an absolute filename.
                    path = strdup(ind);
                } else {
                    continue;
                }
            else
                asprintf(&path, "%s/%s", sl_get(backend->index_paths, j), ind);

            logverb("Trying path %s...\n", path);
            if (index_is_file_index(path)) {
                if (add_index(backend, path))
                    logmsg("Failed to add index \"%s\".\n", path);
                else {
                    found = TRUE;
                    free(path);
                    break;
                }
            }
            free(path);
        }
        if (!found) {
            logmsg("Couldn't find index \"%s\".\n", ind);
            rtn = -1;
            goto done;
        }
    }

    if (auto_index) {
        // Search the paths specified and add any indexes that are found.
        for (i=0; i<sl_size(backend->index_paths); i++) {
            char* path = sl_get(backend->index_paths, i);
            DIR* dir = opendir(path);
            sl* tryinds;
            int j;
            if (!dir) {
                SYSERROR("Failed to open directory \"%s\"", path);
                continue;
            }
            logverb("Auto-indexing directory \"%s\" ...\n", path);
            tryinds = sl_new(16);
            while (1) {
                struct dirent* de;
                char* name;
                char* fullpath;
                char* err;
                bool ok;
                errno = 0;
                de = readdir(dir);
                if (!de) {
                    if (errno)
                        SYSERROR("Failed to read entry from directory \"%s\"", path);
                    break;
                }
                name = de->d_name;

                asprintf(&fullpath, "%s/%s", path, name);

                logverb("\nChecking file \"%s\"\n", fullpath);
                errors_start_logging_to_string();
                ok = index_is_file_index(fullpath);
                err = errors_stop_logging_to_string(": ");
                if (!ok) {
                    logverb("File is not an index: %s\n", err);
                    free(fullpath);
                    continue;
                }

                sl_insert_sorted_nocopy(tryinds, fullpath);
            }
            closedir(dir);

            // in reverse order...
            for (j=sl_size(tryinds)-1; j>=0; j--) {
                char* path = sl_get(tryinds, j);
                logverb("Trying to add index \"%s\".\n", path);
                if (add_index(backend, path)) {
                    logmsg("Failed to add index \"%s\".\n", path);
                }
            }
            sl_free2(tryinds);
        }
    }

 done:
    sl_free2(indices);
	return rtn;
}

struct job_t {
	dl* scales;
	il* depths;
    bool include_default_scales;

    double quad_sizefraction_min;
    double quad_sizefraction_max;

    blind_t bp;
};
typedef struct job_t job_t;

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

static void job_free(job_t* job) {
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

static int run_job(job_t* job, backend_t* backend) {
    blind_t* bp = &(job->bp);
    solver_t* sp = &(bp->solver);

    int i;
    double app_min_default;
    double app_max_default;
    bool solved = FALSE;

    if (blind_is_run_obsolete(bp, sp)) {
        solved = TRUE;
        return 0;
    }

    app_min_default = deg2arcsec(backend->minwidth) / job_imagew(job);
    app_max_default = deg2arcsec(backend->maxwidth) / job_imagew(job);

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
        //logmsg("Feeding to blind: startobj %i, endobj %i\n", startobj, endobj);

		for (j=0; j<dl_size(job->scales) / 2; j++) {
			double fmin, fmax;
			double app_max, app_min;
			int nused;
            int k;

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
            sp->quadsize_min = job->quad_sizefraction_min * MIN(job_imagew(job), job_imageh(job));

			// range of quad sizes that could be found in the field,
			// in arcsec.
            // the hypotenuse...
			fmax = job->quad_sizefraction_max * hypot(job_imagew(job), job_imageh(job)) * app_max;
			fmin = sp->quadsize_min * app_min;

			// Select the indices that should be checked.
			nused = 0;
			for (k = 0; k < bl_size(backend->indexinfos); k++) {
				indexinfo_t* ii = bl_access(backend->indexinfos, k);
				if ((fmin > ii->hisize) || (fmax < ii->losize))
					continue;
                blind_add_index(bp, ii->indexname);
				nused++;
			}

			// Use the (list of) smallest or largest indices if no other one fits.
			if (!nused) {
                il* list;
                if (fmin > backend->sizebiggest) {
                    list = backend->ibiggest;
                } else if (fmax < backend->sizesmallest) {
                    list = backend->ismallest;
                } else {
                    assert(0);
                }
                for (k=0; k<il_size(list); k++) {
                    indexinfo_t* ii;
                    ii = bl_access(backend->indexinfos, il_get(list, k));
                    blind_add_index(bp, ii->indexname);
                }
            }

			if (backend->inparallel)
                bp->indexes_inparallel = TRUE;

            logverb("Running blind");
            blind_log_run_parameters(bp);

            blind_run(bp);

            // we only want to try using the verify_wcses the first time.
            blind_clear_verify_wcses(bp);
            blind_clear_indexes(bp);
            blind_clear_solutions(bp);
            solver_clear_indexes(sp);

            if (blind_is_run_obsolete(bp, sp)) {
                solved = TRUE;
                break;
            }
		}
        if (solved)
            break;
	}

	return 0;
}

bool parse_job_from_qfits_header(qfits_header* hdr, job_t* job) {
    blind_t* bp = &(job->bp);
    solver_t* sp = &(bp->solver);

	double dnil = -HUGE_VAL;
	char *pstr;
	int n;
    bool run;

    double default_poserr = 1.0;
    bool default_tweak = TRUE;
    int default_tweakorder = 2;
    double default_odds_toprint = 1e6;
    double default_odds_tokeep = 1e9;
    double default_odds_tosolve = 1e9;
    //double default_image_fraction = 1.0;
    double default_codetol = 0.01;
	double default_distractor_fraction = 0.25;
    double default_quadsizefraction_min = 0.1;
    double default_quadsizefraction_max = 1.0;
    char* fn;

    blind_init(bp);
    // must be in this order because init_parameters handily zeros out sp
    solver_set_default_values(sp);

    bp->quiet = !verbose;
    bp->verbose = verbose;

    sp->field_maxx = qfits_header_getdouble(hdr, "IMAGEW", dnil);
    sp->field_maxy = qfits_header_getdouble(hdr, "IMAGEH", dnil);
	if ((sp->field_maxx == dnil) || (sp->field_maxy == dnil) ||
		(sp->field_maxx <= 0.0) || (sp->field_maxy <= 0.0)) {
		logerr("Must specify positive \"IMAGEW\" and \"IMAGEH\".\n");
		goto bailout;
	}

    sp->verify_pix = qfits_header_getdouble(hdr, "ANPOSERR", default_poserr);
    sp->codetol    = qfits_header_getdouble(hdr, "ANCTOL",   default_codetol);
    sp->distractor_ratio = qfits_header_getdouble(hdr, "ANDISTR", default_distractor_fraction);
    sp->logratio_bail_threshold = log(1e-100);

    blind_set_solved_file  (bp, fn=fits_get_long_string(hdr, "ANSOLVED"));
    free(fn);
    fn = fits_get_long_string(hdr, "ANSOLVIN");
    if (fn)
        // only set the input solved filename if it was set.
        blind_set_solvedin_file(bp, fn);
    free(fn);
    blind_set_match_file   (bp, fn=fits_get_long_string(hdr, "ANMATCH" ));
    free(fn);
    blind_set_rdls_file    (bp, fn=fits_get_long_string(hdr, "ANRDLS"  ));
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
    bp->cpulimit = qfits_header_getint(hdr, "ANCLIM", 0);
    bp->logratio_toprint = log(qfits_header_getdouble(hdr, "ANODDSPR", default_odds_toprint));
    bp->logratio_tokeep = log(qfits_header_getdouble(hdr, "ANODDSKP", default_odds_tokeep));
    bp->logratio_tosolve = log(qfits_header_getdouble(hdr, "ANODDSSL", default_odds_tosolve));
    bp->best_hit_only = TRUE;

	// job->image_fraction = qfits_header_getdouble(hdr, "ANIMFRAC", job->image_fraction);
    job->include_default_scales = qfits_header_getboolean(hdr, "ANAPPDEF", 0);

    sp->parity = PARITY_BOTH;
	pstr = qfits_pretty_string(qfits_header_getstr(hdr, "ANPARITY"));
	if (pstr && !strcmp(pstr, "NEG"))
		sp->parity = PARITY_FLIP;
	else if (pstr && !strcmp(pstr, "POS"))
		sp->parity = PARITY_NORMAL;

    if (qfits_header_getboolean(hdr, "ANTWEAK", default_tweak)) {
        int order = qfits_header_getint(hdr, "ANTWEAKO", default_tweakorder);
        bp->do_tweak = TRUE;
        bp->tweak_aborder = order;
        bp->tweak_abporder = order;
        bp->tweak_skipshift = TRUE;
    }

    job->quad_sizefraction_min = qfits_header_getdouble(hdr, "ANQSFMIN",
                                                        default_quadsizefraction_min);
    job->quad_sizefraction_max = qfits_header_getdouble(hdr, "ANQSFMAX",
                                                        default_quadsizefraction_max);

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
		int i, j;
		int bail = 0;
        int order;
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
        sprintf(key, "ANW%iSAO", n);
        order = qfits_header_getint(hdr, key, -1);
        if (order >= 2) {
            if (order > 9)
                order = 9;
            wcs.a_order = order;
            wcs.b_order = order;
            for (i=0; i<=order; i++) {
                for (j=0; (i+j)<=order; j++) {
                    if (i+j < 1)
                        continue;
                    sprintf(key, "ANW%iA%i%i", n, i, j);
                    wcs.a[i][j] = qfits_header_getdouble(hdr, key, 0.0);
                    sprintf(key, "ANW%iB%i%i", n, i, j);
                    wcs.b[i][j] = qfits_header_getdouble(hdr, key, 0.0);
                }
            }
        }
        sprintf(key, "ANW%iSAPO", n);
        order = qfits_header_getint(hdr, key, -1);
        if (order >= 2) {
            if (order > 9)
                order = 9;
            wcs.ap_order = order;
            wcs.bp_order = order;
            for (i=0; i<=order; i++) {
                for (j=0; (i+j)<=order; j++) {
                    if (i+j < 1)
                        continue;
                    sprintf(key, "ANW%iAP%i%i", n, i, j);
                    wcs.ap[i][j] = qfits_header_getdouble(hdr, key, 0.0);
                    sprintf(key, "ANW%iBP%i%i", n, i, j);
                    wcs.bp[i][j] = qfits_header_getdouble(hdr, key, 0.0);
                }
            }
        }

        blind_add_verify_wcs(bp, &wcs);
		n++;
	}

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
	backend->indexinfos = bl_new(16, sizeof(indexinfo_t));
	backend->index_paths = sl_new(10);
	backend->ismallest = il_new(4);
	backend->ibiggest = il_new(4);
	backend->default_depths = il_new(4);
	backend->sizesmallest = HUGE_VAL;
	backend->sizebiggest = -HUGE_VAL;

	// Default scale estimate: field width, in degrees:
	backend->minwidth = 0.1;
	backend->maxwidth = 180.0;
    backend->cpulimit = 600;
	return backend;
}

void backend_free(backend_t* backend) {
	int i;
    if (!backend)
        return;
	if (backend->indexinfos) {
		for (i = 0; i < bl_size(backend->indexinfos); i++) {
			indexinfo_t* ii = bl_access(backend->indexinfos, i);
			free(ii->indexname);
		}
		bl_free(backend->indexinfos);
	}
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

int main(int argc, char** args) {
    char* default_configfn = "backend.cfg";
    char* default_config_path = "../etc";

	int c;
	char* configfn = NULL;
	FILE* fconf;
	int i;
	backend_t* backend;
    char* mydir = NULL;
    char* me;
    bool help = FALSE;
    sl* strings = sl_new(4);
    char* cancelfn = NULL;
    int loglvl = LOG_MSG;
    bool tostderr = FALSE;

	while (1) {
		int option_index = 0;
		c = getopt_long(argc, args, OPTIONS, long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
        case 'E':
            tostderr = TRUE;
            break;
		case 'h':
            help = TRUE;
			break;
        case 'v':
            loglvl++;
            break;
        case 'C':
            cancelfn = optarg;
            break;
		case 'c':
			configfn = strdup(optarg);
			break;
		case '?':
			break;
		default:
            printf("Unknown flag %c\n", c);
			exit( -1);
		}
	}

	if (optind == argc) {
		// Need extra args: filename
		printf("You must specify at least one input file!\n\n");
		help = TRUE;
	}
	if (help) {
		print_help(args[0]);
		exit(0);
	}

    log_init(loglvl);
    if (tostderr)
        log_to(stderr);

	backend = backend_new();

    // directory containing the 'backend' executable:
    me = find_executable(args[0], NULL);
    if (!me)
        me = strdup(args[0]);
    mydir = sl_append(strings, dirname(me));
    free(me);

	// Read config file
    if (!configfn) {
        int i;
        sl* trycf = sl_new(4);
        sl_appendf(trycf, "%s/%s/%s", mydir, default_config_path, default_configfn);
        sl_appendf(trycf, "%s/%s", mydir, default_configfn);
        sl_appendf(trycf, "./%s", default_configfn);
        sl_appendf(trycf, "./%s/%s", default_config_path, default_configfn);
        for (i=0; i<sl_size(trycf); i++) {
            char* cf = sl_get(trycf, i);
            if (file_exists(cf)) {
                configfn = strdup(cf);
                logverb("Using config file \"%s\"\n", cf);
                break;
            } else {
                logverb("Config file \"%s\" doesn't exist.\n", cf);
            }
        }
        if (!configfn) {
            char* cflist = sl_join(trycf, "\n  ");
            logerr("Couldn't find config file: tried:\n  %s\n", cflist);
            free(cflist);
        }
        sl_free2(trycf);
    }
	fconf = fopen(configfn, "r");
	if (!fconf) {
		SYSERROR("Failed to open config file \"%s\"", configfn);
		exit( -1);
	}

	if (parse_config_file(fconf, backend)) {
        logerr("Failed to parse config file \"%s\"\n", configfn);
		exit( -1);
	}
	fclose(fconf);

	if (!pl_size(backend->indexinfos)) {
		logerr("You must list at least one index in the config file (%s)\n", configfn);
		exit( -1);
	}

	if (backend->minwidth <= 0.0 || backend->maxwidth <= 0.0) {
		logerr("\"minwidth\" and \"maxwidth\" in the config file %s must be positive!\n", configfn);
		exit( -1);
	}

    free(configfn);

    if (!il_size(backend->default_depths)) {
        parse_depth_string(backend->default_depths,
                           "10 20 30 40 50 60 70 80 90 100 "
                           "110 120 130 140 150 160 170 180 190 200");
    }

	for (i = optind; i < argc; i++) {
		char* jobfn;
		qfits_header* hdr;
		job_t* job;
        blind_t* bp;

		jobfn = args[i];

        logverb("Reading job file \"%s\"...\n", jobfn);

		// Read primary header.
		hdr = qfits_header_read(jobfn);
		if (!hdr) {
			ERROR("Failed to parse FITS header from file \"%s\"", jobfn);
			exit( -1);
		}
        job = job_new();
		if (!parse_job_from_qfits_header(hdr, job)) {
            continue;
        }
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
        if (!bp->cpulimit || bp->cpulimit > backend->cpulimit) {
            logverb("Decreasing CPU time limit to the backend's limit of %i\n",
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

		qfits_header_destroy(hdr);

        if (cancelfn)
            blind_set_cancel_file(bp, cancelfn);

		if (run_job(job, backend))
			logerr("Failed to run_job()\n");

        solver_cleanup(&(bp->solver));
        blind_cleanup(bp);

		job_free(job);
	}

	backend_free(backend);
    sl_free2(strings);

    return 0;
}
