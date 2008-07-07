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

#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "scriptutils.h"
#include "gnu-specific.h"

#include "qfits.h"

static bool verbose = FALSE;

static struct option long_options[] =
    {
	    {"help",    no_argument,       0, 'h'},
        {"verbose", no_argument,       0, 'v'},
	    {"config",  required_argument, 0, 'c'},
	    {"cancel",  required_argument, 0, 'C'},
	    {"input",   required_argument, 0, 'i'},
	    {0, 0, 0, 0}
    };

static const char* OPTIONS = "hc:i:vC:";

static void print_help(const char* progname)
{
	printf("Usage:   %s [options] <augmented xylist>\n"
	       "   [-c <backend config file>]  (default: \"backend.cfg\" in the directory ../etc/ relative to the directory containing the \"backend\" executable)\n"
	       "   [-i <blind input filename>]: save the input file used for blind.\n"
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
	char* blind;
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
        } else if (is_word(line, "blind ", &nextword)) {
			free(backend->blind);
			backend->blind = strdup(nextword);
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

struct job_t
{
	char* fieldfile;
	double imagew;
	double imageh;
	bool run;
	double poserr;
	char* solvedfile;
	char* solvedinfile;
	char* matchfile;
	char* rdlsfile;
	char* wcsfile;
	char* cancelfile;
	int timelimit;
	int cpulimit;
	int parity;
	bool tweak;
	int tweakorder;
	dl* scales;
	il* depths;
	il* fields;
	double odds_toprint;
	double odds_tokeep;
	double odds_tosolve;
	double image_fraction;
	double codetol;
	double distractor_fraction;
	// Contains sip_t structs.
	bl* verify_wcs;
    bool include_default_scales;
    char* xcol;
    char* ycol;
};
typedef struct job_t job_t;

static job_t* job_new()
{
	job_t* job = calloc(1, sizeof(job_t));
	if (!job) {
		printf("Failed to allocate a new job_t.\n");
		return NULL;
	}
	// Default values:
	job->poserr = 1.0;
	job->parity = PARITY_BOTH;
	job->tweak = TRUE;
	job->tweakorder = 3;
	job->scales = dl_new(8);
	job->depths = il_new(8);
	job->fields = il_new(32);
	job->odds_toprint = 1e3;
	job->odds_tokeep = 1e9;
	job->odds_tosolve = 1e9;
	job->image_fraction = 1.0;
	job->codetol = 0.01;
	job->distractor_fraction = 0.25;
	job->verify_wcs = bl_new(8, sizeof(sip_t));

	return job;
}

static void job_free(job_t* job)
{
	if (!job)
		return ;
	free(job->solvedfile);
	free(job->solvedinfile);
	free(job->matchfile);
	free(job->rdlsfile);
	free(job->wcsfile);
	free(job->cancelfile);
	dl_free(job->scales);
	il_free(job->depths);
	il_free(job->fields);
	bl_free(job->verify_wcs);
    free(job->xcol);
    free(job->ycol);
	free(job);
}

static void job_print(job_t* job)
{
	int i;
	printf("Image size: %g x %g\n", job->imagew, job->imageh);
	printf("Positional error: %g pix\n", job->poserr);
	printf("Solved file: %s\n", job->solvedfile);
    if (job->solvedinfile)
        printf("Solved input file: %s\n", job->solvedinfile);
	printf("Match file: %s\n", job->matchfile);
	printf("RDLS file: %s\n", job->rdlsfile);
	printf("WCS file: %s\n", job->wcsfile);
	printf("Cancel file: %s\n", job->cancelfile);
    if (job->xcol)
        printf("X column: %s\n", job->xcol);
    if (job->ycol)
        printf("Y column: %s\n", job->ycol);
	printf("Time limit: %i sec\n", job->timelimit);
	printf("CPU limit: %i sec\n", job->cpulimit);
	printf("Parity: %s\n", (job->parity == PARITY_NORMAL ? "pos" :
	                        (job->parity == PARITY_FLIP ? "neg" :
	                         (job->parity == PARITY_BOTH ? "both" : "(unknown)"))));
	printf("Tweak: %s\n", (job->tweak ? "yes" : "no"));
	printf("Tweak order: %i\n", job->tweakorder);
	printf("Odds to print: %g\n", job->odds_toprint);
	printf("Odds to keep: %g\n", job->odds_tokeep);
	printf("Odds to solve: %g\n", job->odds_tosolve);
	printf("Image fraction: %g\n", job->image_fraction);
	printf("Distractor fraction: %g\n", job->distractor_fraction);
	printf("Code tolerance: %g\n", job->codetol);
	printf("Scale ranges:\n");
	for (i = 0; i < dl_size(job->scales) / 2; i++) {
		double lo, hi;
		lo = dl_get(job->scales, i * 2);
		hi = dl_get(job->scales, i * 2 + 1);
		printf("  [%g, %g] arcsec/pix\n", lo, hi);
	}
	printf("Depths:");
	for (i = 0; i < il_size(job->depths)/2; i++) {
		int dlo, dhi;
        dlo = il_get(job->depths, 2*i);
        dhi = il_get(job->depths, 2*i + 1);
        if (!dhi)
            printf(" %i-", dlo);
        else
            printf(" %i-%i", dlo, dhi);
	}
	printf("\n");
	printf("Fields:");
	for (i = 0; i < il_size(job->fields) / 2; i++) {
		int lo, hi;
		lo = il_get(job->fields, i * 2);
		hi = il_get(job->fields, i * 2 + 1);
		if (lo == hi)
			printf(" %i", lo);
		else
			printf(" %i-%i", lo, hi);
	}
	printf("\n");
	printf("Verify WCS:\n");
	for (i = 0; i < bl_size(job->verify_wcs); i++) {
		sip_t* wcs = bl_access(job->verify_wcs, i);
		printf("  crpix (%g, %g)\n", wcs->wcstan.crpix[0], wcs->wcstan.crpix[1]);
		printf("  crval (%g, %g)\n", wcs->wcstan.crval[0], wcs->wcstan.crval[1]);
		printf("  cd  = ( %g, %g )\n", wcs->wcstan.cd[0][0], wcs->wcstan.cd[0][1]);
		printf("        ( %g, %g )\n", wcs->wcstan.cd[1][0], wcs->wcstan.cd[1][1]);
	}
	printf("Run: %s\n", (job->run ? "yes" : "no"));
}

#define WRITE(f, x, ...) \
do {\
  if (fprintf(f, x, ##__VA_ARGS__) < 0) { \
    fprintf(stderr, "Failed to write: %s\n", strerror(errno)); \
    return -1; \
  } \
} while(0)

static int job_write_blind_input(job_t* job, FILE* fout, backend_t* backend)
{
	int i, j, k;
    il* depths;
    il* nolimit;
    double app_min_default;
    double app_max_default;
	bool firsttime = TRUE;

    app_min_default = deg2arcsec(backend->minwidth) / job->imagew;
    app_max_default = deg2arcsec(backend->maxwidth) / job->imagew;

    if (job->timelimit)
        WRITE(fout, "timelimit %i\n", job->timelimit);
    if (job->cpulimit)
        WRITE(fout, "cpulimit %i\n", job->cpulimit);

    nolimit = il_new(4);
    il_append(nolimit, 0);
    il_append(nolimit, 0);

    if (il_size(job->depths))
        depths = job->depths;
    else {
        if (backend->inparallel)
            depths = nolimit;
        else
            depths = backend->default_depths;
    }

    for (i=0; i<il_size(depths)/2; i++) {
		int startobj = il_get(depths, i*2);
        int endobj = il_get(depths, i*2+1);

        // make depth ranges be inclusive.
        if (startobj || endobj) {
            endobj++;
        }

		for (j = 0; j < dl_size(job->scales) / 2; j++) {
			double fmin, fmax;
			double app_max, app_min;
			int nused;

			WRITE(fout, "sdepth %i\n", startobj);
			if (endobj)
				WRITE(fout, "depth %i\n", endobj);
			// arcsec per pixel range
			app_min = dl_get(job->scales, j * 2);
			app_max = dl_get(job->scales, j * 2 + 1);
            if (app_min == 0.0)
                app_min = app_min_default;
            WRITE(fout, "fieldunits_lower %g\n", app_min);
            if (app_max == 0.0)
                app_max = app_max_default;
            WRITE(fout, "fieldunits_upper %g\n", app_max);

			WRITE(fout, "fieldw %g\n", job->imagew);
			WRITE(fout, "fieldh %g\n", job->imageh);

			// minimum quad size to try (in pixels)
			WRITE(fout, "quadsize_min %g\n", 0.1 * MIN(job->imagew, job->imageh));

			// range of quad sizes that could be found in the field,
			// in arcsec.
			fmax = 1.0 * MAX(job->imagew, job->imageh) * app_max;
			fmin = 0.1 * MIN(job->imagew, job->imageh) * app_min;

			// Select the indices that should be checked.
			nused = 0;
			for (k = 0; k < bl_size(backend->indexinfos); k++) {
				indexinfo_t* ii = bl_access(backend->indexinfos, k);
				if ((fmin > ii->hisize) || (fmax < ii->losize))
					continue;
				WRITE(fout, "index %s\n", ii->indexname);
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
                    WRITE(fout, "index %s\n", ii->indexname);
                }
            }
			if (backend->inparallel)
				WRITE(fout, "indexes_inparallel\n");

			WRITE(fout, "fields");
			for (k = 0; k < il_size(job->fields) / 2; k++) {
				int lo = il_get(job->fields, k * 2);
				int hi = il_get(job->fields, k * 2 + 1);
				if (lo == hi)
					WRITE(fout, " %i", lo);
				else
					WRITE(fout, " %i/%i", lo, hi);
			}
			WRITE(fout, "\n");

			WRITE(fout, "parity %i\n", job->parity);
			WRITE(fout, "verify_pix %g\n", job->poserr);
			WRITE(fout, "tol %g\n", job->codetol);
			WRITE(fout, "distractors %g\n", job->distractor_fraction);
			WRITE(fout, "ratio_toprint %g\n", job->odds_toprint);
			WRITE(fout, "ratio_tokeep %g\n", job->odds_tokeep);
			WRITE(fout, "ratio_tosolve %g\n", job->odds_tosolve);
			WRITE(fout, "ratio_tobail %g\n", 1e-100);
			WRITE(fout, "best_only\n");

            if (job->xcol)
                WRITE(fout, "xcol %s\n", job->xcol);
            if (job->ycol)
                WRITE(fout, "ycol %s\n", job->ycol);

			if (job->tweak) {
				WRITE(fout, "tweak\n");
				WRITE(fout, "tweak_aborder %i\n", job->tweakorder);
				WRITE(fout, "tweak_abporder %i\n", job->tweakorder);
				WRITE(fout, "tweak_skipshift\n");
			}

			WRITE(fout, "field %s\n", job->fieldfile);
			if (job->solvedfile)
				WRITE(fout, "solved %s\n", job->solvedfile);
			if (job->solvedinfile)
				WRITE(fout, "solved_in %s\n", job->solvedinfile);
			if (job->matchfile)
				WRITE(fout, "match %s\n", job->matchfile);
			if (job->rdlsfile) {
				WRITE(fout, "indexrdls %s\n", job->rdlsfile);
				//WRITE(fout, "indexrdls_solvedonly\n");
            }
			if (job->wcsfile)
				WRITE(fout, "wcs %s\n", job->wcsfile);
			if (job->cancelfile)
				WRITE(fout, "cancel %s\n", job->cancelfile);

			if (firsttime) {
				for (k = 0; k < bl_size(job->verify_wcs); k++) {
					sip_t* wcs = bl_access(job->verify_wcs, k);
					WRITE(fout, "verify_wcs %g %g %g %g %g %g %g %g",
					      wcs->wcstan.crval[0], wcs->wcstan.crval[1],
					      wcs->wcstan.crpix[0], wcs->wcstan.crpix[1],
					      wcs->wcstan.cd[0][0], wcs->wcstan.cd[0][1],
					      wcs->wcstan.cd[1][0], wcs->wcstan.cd[1][1]);
                    if (wcs->a_order || wcs->ap_order) {
                        int m, n;
                        assert(wcs->a_order == wcs->b_order);
                        assert(wcs->ap_order == wcs->bp_order);
                        WRITE(fout, " %i %i", wcs->a_order, wcs->ap_order);
                        for (m=0; m<=wcs->a_order; m++) {
                            for (n=0; (m+n)<=wcs->a_order; n++) {
                                WRITE(fout, " %g %g", wcs->a[m][n], wcs->b[m][n]);
                            }
                        }
                        for (m=0; m<=wcs->ap_order; m++) {
                            for (n=0; (m+n)<=wcs->ap_order; n++) {
                                WRITE(fout, " %g %g", wcs->ap[m][n], wcs->bp[m][n]);
                            }
                        }
                    }
                    WRITE(fout, "\n");
				}
				firsttime = FALSE;
			}

			WRITE(fout, "run\n\n");
		}

	}

    il_free(nolimit);

	return 0;
}

static int run_blind(job_t* job, backend_t* backend)
{
	int thepipe[2];
	pid_t pid;

	if (pipe(thepipe) == -1) {
		fprintf(stderr, "Error creating pipe: %s\n", strerror(errno));
		return -1;
	}

	fflush(stdout);
	fflush(stderr);

	pid = fork();
	if (pid == -1) {
		fprintf(stderr, "Error fork()ing: %s\n", strerror(errno));
		return -1;
	} else if (pid == 0) {
		int old_stdin;
		// Child process.
		close(thepipe[1]);
		// bind stdin to thepipe[0].
		old_stdin = dup(STDIN_FILENO);
		if (old_stdin == -1) {
			fprintf(stderr, "Failed to save stdin: %s\n", strerror(errno));
			_exit( -1);
		}
		if (dup2(thepipe[0], STDIN_FILENO) == -1) {
			fprintf(stderr, "Failed to dup2 stdin: %s\n", strerror(errno));
			_exit( -1);
		}

		// Use a "system"-like command to allow fancier "blind" commands.
		if (execlp("/bin/sh", "/bin/sh", "-c", backend->blind, (char*)NULL)) {    

			fprintf(stderr, "Failed to execlp blind: %s\n", strerror(errno));
			_exit( -1);
		}
		// execlp doesn't return.
	} else {
		FILE* fpipe;
		int status;
		// Parent process.
		close(thepipe[0]);
		fpipe = fdopen(thepipe[1], "a");
		if (!fpipe) {
			fprintf(stderr, "Failed to fdopen pipe: %s\n", strerror(errno));
			return -1;
		}
		// Write input to blind.
		if (job_write_blind_input(job, fpipe, backend)) {
			fprintf(stderr, "Failed to write input file to blind: %s\n", strerror(errno));
			return -1;
		}
		fclose(fpipe);

		// Wait for blind to finish.
        if (verbose)
            printf("Waiting for blind to finish (PID %i).\n", (int)pid);
		do {
			if (waitpid(pid, &status, 0) == -1) {
				fprintf(stderr, "Failed to waitpid() for blind: %s.\n", strerror(errno));
				return -1;
			}
			if (WIFSIGNALED(status)) {
				// (WTERMSIG(status) == SIGINT || WTERMSIG(status) == SIGQUIT)) {
				//fprintf(stderr, "Blind was killed.\n");
				fprintf(stderr, "Blind was killed by signal %i.\n", WTERMSIG(status));
				return -1;
			} else {
				int exitval = WEXITSTATUS(status);
				if (exitval == 127) {
					fprintf(stderr, "Blind executable not found.\n");
					return -1;
				} else if (exitval) {
					fprintf(stderr, "Blind executable failed: return value %i.\n", exitval);
					return -1;
				}
			}
		} while (!WIFEXITED(status) && !WIFSIGNALED(status));
        if (verbose)
            printf("Blind finished successfully.\n");
	}
	return 0;
}

job_t* parse_job_from_qfits_header(qfits_header* hdr)
{
	double dnil = -HUGE_VAL;
	job_t* job = job_new();
	char *pstr;
	int n;

	job->imagew = qfits_header_getdouble(hdr, "IMAGEW", dnil);
	job->imageh = qfits_header_getdouble(hdr, "IMAGEH", dnil);
	if ((job->imagew == dnil) || (job->imageh == dnil) ||
		(job->imagew <= 0.0) || (job->imageh <= 0.0)) {
		printf("Must specify positive \"IMAGEW\" and \"IMAGEH\".\n");
		goto bailout;
	}
	job->run = qfits_header_getboolean(hdr, "ANRUN", 0);
	job->poserr = qfits_header_getdouble(hdr, "ANPOSERR", job->poserr);
	job->solvedfile = fits_get_dupstring(hdr, "ANSOLVED");
	job->solvedinfile = fits_get_dupstring(hdr, "ANSOLVIN");
	job->matchfile = fits_get_dupstring(hdr, "ANMATCH");
	job->rdlsfile = fits_get_dupstring(hdr, "ANRDLS");
	job->wcsfile = fits_get_dupstring(hdr, "ANWCS");
	job->cancelfile = fits_get_dupstring(hdr, "ANCANCEL");
	job->timelimit = qfits_header_getint(hdr, "ANTLIM", job->timelimit);
	job->cpulimit = qfits_header_getint(hdr, "ANCLIM", job->cpulimit);
    job->include_default_scales = qfits_header_getboolean(hdr, "ANAPPDEF", 0);

	job->xcol = fits_get_dupstring(hdr, "ANXCOL");
	job->ycol = fits_get_dupstring(hdr, "ANYCOL");

	pstr = qfits_pretty_string(qfits_header_getstr(hdr, "ANPARITY"));
	if (pstr && !strcmp(pstr, "NEG")) {
		job->parity = PARITY_FLIP;
	} else if (pstr && !strcmp(pstr, "POS")) {
		job->parity = PARITY_NORMAL;
	}
	job->tweak = qfits_header_getboolean(hdr, "ANTWEAK", job->tweak);
	job->tweakorder = qfits_header_getint(hdr, "ANTWEAKO", job->tweakorder);
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
		il_append(job->fields, lo);
		il_append(job->fields, hi);
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
		il_append(job->fields, fld);
		il_append(job->fields, fld);
		n++;
	}
	job->odds_toprint = qfits_header_getdouble(hdr, "ANODDSPR", job->odds_toprint);
	job->odds_tokeep = qfits_header_getdouble(hdr, "ANODDSKP", job->odds_tokeep);
	job->odds_tosolve = qfits_header_getdouble(hdr, "ANODDSSL", job->odds_tosolve);
	job->image_fraction = qfits_header_getdouble(hdr, "ANIMFRAC", job->image_fraction);
	job->codetol = qfits_header_getdouble(hdr, "ANCTOL", job->codetol);
	job->distractor_fraction = qfits_header_getdouble(hdr, "ANDISTR", job->distractor_fraction);
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

		bl_append(job->verify_wcs, &wcs);
		n++;
	}

	// Default: solve first field.
	if (job->run && !il_size(job->fields)) {
		il_append(job->fields, 1);
		il_append(job->fields, 1);
	}

	return job;

 bailout:
	job_free(job);
	return NULL;
}

backend_t* backend_new()
{
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

void backend_free(backend_t* backend)
{
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
	if (backend->blind)
		free(backend->blind);
    free(backend);
}

int main(int argc, char** args)
{
    char* default_configfn = "backend.cfg";
    char* default_config_path = "../etc";
    char* default_blind_command = "blind";

	int c;
	char* configfn = NULL;
	char* inputfn = NULL;
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
		case 'i':
			inputfn = optarg;
			break;
		case '?':
			break;
		default:
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

    if (!backend->blind) {
        // default "blind": relative to backend.
        char* blindcmd;
        asprintf_safe(&blindcmd, "%s/%s", mydir, default_blind_command);
        backend->blind = blindcmd;
    }

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
		job_t* job = NULL;

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
		job = parse_job_from_qfits_header(hdr);
		job->fieldfile = jobfn;

		// If the job has no scale estimate, search everything provided
		// by the backend
		if (!dl_size(job->scales) || job->include_default_scales) {
			double arcsecperpix;
			arcsecperpix = deg2arcsec(backend->minwidth) / job->imagew;
			dl_append(job->scales, arcsecperpix);
			arcsecperpix = deg2arcsec(backend->maxwidth) / job->imagew;
			dl_append(job->scales, arcsecperpix);
		}

        // The job can only decrease the CPU limit.
        if (!job->cpulimit || job->cpulimit > backend->cpulimit) {
            job->cpulimit = backend->cpulimit;
        }

		qfits_header_destroy(hdr);

        if (cancelfn) {
            job->cancelfile = strdup(cancelfn);
        }

        if (verbose) {
            printf("Running job:\n");
            job_print(job);
            printf("\n");
            printf("Input file for blind:\n\n");
            job_write_blind_input(job, stdout, backend);
        }

		if (inputfn) {
			FILE* f = fopen(inputfn, "a");
			if (!f) {
				fprintf(stderr, "Failed to open file \"%s\" to save the input sent to blind: %s.\n",
				        inputfn, strerror(errno));
				exit( -1);
			}
			if (job_write_blind_input(job, f, backend) ||
			        fclose(f)) {
				fprintf(stderr, "Failed to save the blind input file to \"%s\": %s.\n", inputfn, strerror(errno));
				exit( -1);
			}
		}

		if (run_blind(job, backend)) {
			fprintf(stderr, "Failed to run_blind.\n");
		}

		//cleanup:
		job_free(job);
	}

	backend_free(backend);
    sl_free2(strings);

	exit(0);
}
