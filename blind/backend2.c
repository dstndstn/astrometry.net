/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

static bool verbose = FALSE;

static struct option long_options[] =
    {
	    {"help",    no_argument,       0, 'h'},
        {"verbose", no_argument,       0, 'v'},
	    {"config",  required_argument, 0, 'c'},
	    {"cancel",  required_argument, 0, 'C'},
	    {0, 0, 0, 0}
    };

static const char* OPTIONS = "hc:i:vC:";

static void print_help(const char* progname) {
	printf("Usage:   %s [options] <augmented xylist>\n"
	       "   [-c <backend config file>]  (default: \"backend.cfg\" in the directory ../etc/ relative to the directory containing the \"backend\" executable)\n"
           "   [-C <cancel-filename>]: quit solving if the file <cancel-filename> appears.\n"
           "   [-v]: verbose\n"
	       "\n", progname);
}

struct indexinfo {
	char* indexname;
    char* canonname;
	// quad size
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

static char* get_canonical(const char* indexname) {
    // canonicalize_file_name requires an extant file; "indexname" is the base
    // name, so add the ".quad.fits" suffix.
    char* path;
    char* canon;
    asprintf(&path, "%s.quad.fits", indexname);
    canon = canonicalize_file_name(path);
    free(path);
    return canon;
}

static int get_index_scales(const char* indexname,
                            double* losize, double* hisize)
{
	char *quadfname;
	quadfile* quads;
	double hi, lo;

	quadfname = mk_quadfn(indexname);
    if (verbose)
        printf("Reading quads file %s...\n", quadfname);
	quads = quadfile_open(quadfname);
	if (!quads) {
        if (verbose)
            printf("Couldn't read quads file %s\n", quadfname);
		free_fn(quadfname);
		return -1;
	}
	free_fn(quadfname);
	lo = quadfile_get_index_scale_lower_arcsec(quads);
	hi = quadfile_get_index_scale_upper_arcsec(quads);
	if (losize)
		*losize = lo;
	if (hisize)
		*hisize = hi;
    if (verbose) {
        printf("Stars: %i, Quads: %i.\n", quads->numstars, quads->numquads);
        printf("Index scale: [%g, %g] arcmin, [%g, %g] arcsec\n",
               lo / 60.0, hi / 60.0, lo, hi);
    }
	quadfile_close(quads);
	return 0;
}

static bool add_index(backend_t* backend, char* full_index_path, double lo, double hi) {
	indexinfo_t ii;
    int k;

	ii.indexname = full_index_path;
    ii.canonname = get_canonical(full_index_path);
	ii.losize = lo;
	ii.hisize = hi;

    // check that this canonical path isn't already in the list of
    // index files that we've openend.
    for (k=0; k<bl_size(backend->indexinfos); k++) {
        indexinfo_t* iii = bl_access(backend->indexinfos, k);
        if (strcmp(iii->canonname, ii.canonname))
            continue;
        if (verbose)
            printf("Skipping duplicate index %s\n", full_index_path);
        free(ii.canonname);
        return FALSE;
    }
	bl_append(backend->indexinfos, &ii);
	if (ii.losize < backend->sizesmallest) {
		backend->sizesmallest = ii.losize;
        bl_remove_all(backend->ismallest);
		il_append(backend->ismallest, bl_size(backend->indexinfos) - 1);
	} else if (ii.losize == backend->sizesmallest) {
		il_append(backend->ismallest, bl_size(backend->indexinfos) - 1);
    }
	if (ii.hisize > backend->sizebiggest) {
		backend->sizebiggest = ii.hisize;
        bl_remove_all(backend->ibiggest);
		il_append(backend->ibiggest, bl_size(backend->indexinfos) - 1);
	} else if (ii.hisize == backend->sizebiggest) {
		il_append(backend->ibiggest, bl_size(backend->indexinfos) - 1);
	}
    return TRUE;
}

static int find_index(backend_t* backend, char* index)
{
	bool found_index = FALSE;
	double lo, hi;
	int i = 0;
	char* full_index_path;

    if (strlen(index) && index[0] == '/') {
        // it's an absolute path - don't search dirs.
        full_index_path = strdup(index);
        if (get_index_scales(full_index_path, &lo, &hi) == 0) {
			found_index = TRUE;
		} else {
            free(full_index_path);
        }
    }
    if (!found_index) {
        for (i=0; i<sl_size(backend->index_paths); i++) {
            char* index_path = sl_get(backend->index_paths, i);
            asprintf_safe(&full_index_path, "%s/%s", index_path, index);
            if (get_index_scales(full_index_path, &lo, &hi) == 0) {
                found_index = TRUE;
                break;
            }
            free(full_index_path);
        }
    }
	if (!found_index) {
		printf("Failed to find the index \"%s\".\n", index);
		return -1;
	}
    if (verbose)
        printf("Found index: %s\n", full_index_path);

    if (!add_index(backend, full_index_path, lo, hi)) {
        free(full_index_path);
    }
	return 0;
}

static int parse_config_file(FILE* fconf, backend_t* backend)
{
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
			printf("Failed to read a line from the config file: %s\n", strerror(errno));
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
            int i;
            //
            if (parse_positive_range_string(backend->default_depths, nextword, 0, 0, "Depth")) {
                rtn = -1;
                goto done;
            }
            for (i=0; i<il_size(backend->default_depths)/2; i++) {
                int lo, hi;
                lo = il_get(backend->default_depths, 2*i);
                hi = il_get(backend->default_depths, 2*i+1);
                if ((lo == hi) && (2*i+2 < il_size(backend->default_depths))) {
                    // "hi" = the next "lo" - 1
                    hi = il_get(backend->default_depths, 2*i+2) - 1;
                    il_set(backend->default_depths, 2*i+1, hi);
                }
            }
		} else if (is_word(line, "add_path ", &nextword)) {
			sl_append(backend->index_paths, nextword);
		} else {
			printf("Didn't understand this config file line: \"%s\"\n", line);
			// unknown config line is a firing offense
            rtn = -1;
            goto done;
		}
	}

    for (i=0; i<sl_size(indices); i++) {
        char* ind = sl_get(indices, i);
        if (verbose)
            printf("Trying index %s...\n", ind);
        if (find_index(backend, ind)) {
            rtn = -1;
            goto done;
        }
    }

    if (auto_index) {
        // Search the paths specified and add any indexes that are found.
        for (i=0; i<sl_size(backend->index_paths); i++) {
            char* path = sl_get(backend->index_paths, i);
            DIR* dir = opendir(path);
            sl* trybases;
            int j;
            if (!dir) {
                fprintf(stderr, "Failed to open directory \"%s\": %s\n", path, strerror(errno));
                continue;
            }
            if (verbose)
                printf("Auto-indexing directory %s...\n", path);
            trybases = sl_new(16);
            while (1) {
                struct dirent* de;
                char* name;
                int len;
                int baselen;
                char* base;
                errno = 0;
                de = readdir(dir);
                if (!de) {
                    if (errno)
                        fprintf(stderr, "Failed to read entry from directory \"%s\": %s\n", path, strerror(errno));
                    break;
                }
                name = de->d_name;
                // Look for files ending with .quad.fits
                len = strlen(name);
                if (len <= 10)
                    continue;
                baselen = len - 10;
                if (strcmp(name + baselen, ".quad.fits"))
                    continue;
                base = malloc(baselen + 1);
                memcpy(base, name, baselen);
                base[baselen] = '\0';

                sl_insert_sorted(trybases, base);
                free(base);
            }
            // reverse-sort
            for (j=sl_size(trybases)-1; j>=0; j--) {
                char* base = sl_get(trybases, j);
                char* fullpath;
                double lo, hi;
                // FIXME - look for corresponding .skdt.fits and .ckdt.fits files?
                asprintf_safe(&fullpath, "%s/%s", path, base);
                if (verbose)
                    printf("Trying to add index \"%s\".\n", fullpath);
                if (get_index_scales(fullpath, &lo, &hi) == 0) {
                    if (!add_index(backend, fullpath, lo, hi)) {
                        free(fullpath);
                    }
                } else {
                    if (verbose)
                        printf("Failed to add index \"%s\".\n", fullpath);
                    free(fullpath);
                }
            }
            sl_free2(trybases);
            closedir(dir);
        }
    }

 done:
    sl_free2(indices);
	return rtn;
}

struct job_t {
	char* fieldfile;
	dl* scales;
	il* depths;
    bool include_default_scales;

    blind_t bp;
};
typedef struct job_t job_t;

static job_t* job_new() {
	job_t* job = calloc(1, sizeof(job_t));
	if (!job) {
		printf("Failed to allocate a new job_t.\n");
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

    //if (verbose)
    //log_init(4);
    //else
    log_init(3);

    app_min_default = deg2arcsec(backend->minwidth) / job_imagew(job);
    app_max_default = deg2arcsec(backend->maxwidth) / job_imagew(job);

    for (i=0; i<il_size(job->depths)/2; i++) {
		int startobj = il_get(job->depths, i*2);
        int endobj = il_get(job->depths, i*2+1);
        int j;

        // make depth ranges be inclusive.
        if (startobj || endobj) {
            endobj++;
        }

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

            //blind_init(bp);
            // must be in this order because init_parameters handily zeros out sp
            //solver_set_default_values(sp);

            sp->startobj = startobj;
			if (endobj)
                sp->endobj = endobj;

			// minimum quad size to try (in pixels)
            sp->quadsize_min = 0.1 * MIN(job_imagew(job), job_imageh(job));

			// range of quad sizes that could be found in the field,
			// in arcsec.
			fmax = 1.0 * MAX(job_imagew(job), job_imageh(job)) * app_max;
			fmin = 0.1 * MIN(job_imagew(job), job_imageh(job)) * app_min;

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

            blind_set_field_file(bp, job->fieldfile);

            printf("Running blind:\n");
            if (verbose)
                blind_log_run_parameters(bp);

            blind_run(bp);

            // we only want to try using the verify_wcses the first time.
            blind_clear_verify_wcses(bp);
            blind_clear_indexes(bp);
            blind_clear_solutions(bp);
            solver_cleanup(sp);
		}
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

    blind_init(bp);
    // must be in this order because init_parameters handily zeros out sp
    solver_set_default_values(sp);

    bp->quiet = !verbose;
    bp->verbose = verbose;

    sp->field_maxx = qfits_header_getdouble(hdr, "IMAGEW", dnil);
    sp->field_maxy = qfits_header_getdouble(hdr, "IMAGEH", dnil);
	if ((sp->field_maxx == dnil) || (sp->field_maxy == dnil) ||
		(sp->field_maxx <= 0.0) || (sp->field_maxy <= 0.0)) {
		printf("Must specify positive \"IMAGEW\" and \"IMAGEH\".\n");
		goto bailout;
	}

    sp->verify_pix = qfits_header_getdouble(hdr, "ANPOSERR", default_poserr);
    sp->codetol = qfits_header_getdouble(hdr, "ANCTOL", default_codetol);
    sp->distractor_ratio = qfits_header_getdouble(hdr, "ANDISTR", default_distractor_fraction);
    sp->logratio_bail_threshold = 1e-100;

    blind_set_solved_file(bp, fits_get_dupstring(hdr, "ANSOLVED"));
    blind_set_solvedin_file(bp, fits_get_dupstring(hdr, "ANSOLVIN"));
    blind_set_match_file(bp, fits_get_dupstring(hdr, "ANMATCH"));
    blind_set_rdls_file(bp, fits_get_dupstring(hdr, "ANRDLS"));
    blind_set_wcs_file(bp, fits_get_dupstring(hdr, "ANWCS"));
    blind_set_cancel_file(bp, fits_get_dupstring(hdr, "ANCANCEL"));

    blind_set_xcol(bp, fits_get_dupstring(hdr, "ANXCOL"));
    blind_set_ycol(bp, fits_get_dupstring(hdr, "ANYCOL"));

    bp->timelimit = qfits_header_getint(hdr, "ANTLIM", 0);
    bp->cpulimit = qfits_header_getint(hdr, "ANCLIM", 0);
    bp->logratio_toprint = qfits_header_getdouble(hdr, "ANODDSPR", default_odds_toprint);
    bp->logratio_tokeep = qfits_header_getdouble(hdr, "ANODDSKP", default_odds_tokeep);
    bp->logratio_tosolve = qfits_header_getdouble(hdr, "ANODDSSL", default_odds_tosolve);
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
                fprintf(stderr, "Scale range %g to %g is invalid: min must be >= 0, max must be >= min.\n", lo, hi);
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
        if ((dlo < 0) || (dlo > dhi)) {
            fprintf(stderr, "Depth range %i to %i is invalid: min must be >= 1, max must be >= min.\n", dlo, dhi);
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
            fprintf(stderr, "Field range %i to %i is invalid: min must be >= 1, max must be >= min.\n", lo, hi);
            fprintf(stderr, "  (FITS headers: \"%s = %s\", \"%s = %s\")\n",
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
            fprintf(stderr, "Field %i is invalid: must be >= 1.  (FITS header: \"%s = %s\")\n", fld, key,
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
			free(ii->canonname);
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

	while (1) {
		int option_index = 0;
		c = getopt_long(argc, args, OPTIONS, long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
		case 'h':
            help = TRUE;
			break;
        case 'v':
            verbose = TRUE;
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
		printf("You must specify a job file.\n\n");
		help = TRUE;
	}
	if (help) {
		print_help(args[0]);
		exit(0);
	}

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
                if (verbose)
                    printf("Using config file \"%s\"\n", cf);
                break;
            } else {
                if (verbose)
                    printf("Config file \"%s\" doesn't exist.\n", cf);
            }
        }
        if (!configfn) {
            fprintf(stderr, "Couldn't find config file: tried ");
            for (i=0; i<sl_size(trycf); i++) {
                char* cf = sl_get(trycf, i);
                fprintf(stderr, "%s\"%s\"", (i ? ", " : ""), cf);
            }
            fprintf(stderr, "\n");
        }
        sl_free2(trycf);
    }
	fconf = fopen(configfn, "r");
	if (!fconf) {
		fprintf(stderr, "Failed to open config file \"%s\": %s.\n", configfn, strerror(errno));
		exit( -1);
	}

	if (parse_config_file(fconf, backend)) {
		fprintf(stderr, "Failed to parse config file \"%s\".\n", configfn);
		exit( -1);
	}
	fclose(fconf);

	if (!pl_size(backend->indexinfos)) {
		fprintf(stderr, "You must list at least one index in the config file (%s)\n", configfn);
		exit( -1);
	}

	if (backend->minwidth <= 0.0 || backend->maxwidth <= 0.0) {
		fprintf(stderr, "\"minwidth\" and \"maxwidth\" must be positive!\n");
		exit( -1);
	}

    free(configfn);

    if (!il_size(backend->default_depths)) {
        int step = 10;
        int max = 200;
        int i;
        for (i=0; i<max; i+=step) {
            il_append(backend->default_depths, i);
            il_append(backend->default_depths, i+step-1);
        }
    }

	for (i = optind; i < argc; i++) {
		char* jobfn;
		qfits_header* hdr;
		job_t* job;
        blind_t* bp;

		jobfn = args[i];

        if (verbose)
            printf("Reading job file \"%s\"...\n", jobfn);
        fflush(NULL);

		// Read primary header.
		hdr = qfits_header_read(jobfn);
		if (!hdr) {
			fprintf(stderr, "Failed to parse FITS header from file \"%s\".\n", jobfn);
			exit( -1);
		}
        job = job_new();
		if (!parse_job_from_qfits_header(hdr, job)) {
            continue;
        }
		job->fieldfile = jobfn;

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
        bp = &(job->bp);
        if (!bp->cpulimit || bp->cpulimit > backend->cpulimit)
            bp->cpulimit = backend->cpulimit;

        // If the job didn't specify depths, set defaults.
        if (il_size(job->depths) == 0) {
            if (backend->inparallel) {
                // no limit.
                il_append(job->depths, 0);
                il_append(job->depths, 0);
            } else {
                int k;
                for (k=0; k<il_size(backend->default_depths); k++)
                    il_append(job->depths, il_get(backend->default_depths, k));
            }
        }

		qfits_header_destroy(hdr);

        if (cancelfn)
            blind_set_cancel_file(bp, cancelfn);

		if (run_job(job, backend)) {
			fprintf(stderr, "Failed to run_blind.\n");
		}

		//cleanup:
		job_free(job);
	}

	backend_free(backend);
    sl_free2(strings);

    return 0;
}
