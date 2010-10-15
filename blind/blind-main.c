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

#include <ctype.h>
#include <math.h>

#include "qfits_cache.h"
#include "fitsioutils.h"
#include "boilerplate.h"
#include "blind.h"
#include "log.h"
#include "tic.h"

static int read_parameters(blind_t* bp);

int main(int argc, char *argv[]) {
	blind_t my_bp;
	blind_t* bp = &my_bp;
	solver_t* sp = &(bp->solver);

	log_init(LOG_MSG);
    fits_use_error_system();

	if (argc == 2 && strcmp(argv[1], "-s") == 0) {
		log_set_level(LOG_NONE);
		fprintf(stderr, "premptive silence\n");
	}

	// Read input settings until "run" is encountered; repeat.
	for (;;) {
		tic();

		blind_init(bp);
		// must be in this order because init_parameters handily zeros out sp
		solver_set_default_values(sp);

		if (read_parameters(bp)) {
			solver_cleanup(sp);
			blind_cleanup(bp);
			break;
		}

		if (!blind_parameters_are_sane(bp, sp)) {
			exit(-1);
		}

		if (blind_is_run_obsolete(bp, sp)) {
            goto clean;
		}

        blind_log_run_parameters(bp);

		blind_run(bp);

        toc();

		if (bp->hit_total_timelimit)
			break;
		if (bp->hit_total_cpulimit)
			break;

    clean:
		solver_cleanup(sp);
		blind_cleanup(bp);
	}

	qfits_cache_purge(); // for valgrind
	return 0;
}

static int read_parameters(blind_t* bp) {
	solver_t* sp = &(bp->solver);
	for (;;) {
		char buffer[10240];
		char* nextword;
		char* line;
		if (!fgets(buffer, sizeof(buffer), stdin)) {
			return -1;
		}
		line = buffer;
		// strip off newline
		if (line[strlen(line) - 1] == '\n')
			line[strlen(line) - 1] = '\0';

		// skip leading whitespace:
		while (*line && isspace(*line))
			line++;

		logverb("Command: %s\n", line);

		if (line[0] == '#') {
			//logmsg("Skipping comment.\n");
			continue;
		}
		// skip blank lines.
		if (line[0] == '\0') {
			continue;
		}
		if (is_word(line, "help", &nextword)) {
			logmsg("No help soup for you!\n  (use the source, Luke)\n");
        } else if (is_word(line, "logtostderr", &nextword)) {
            log_to(stderr);
        } else if (is_word(line, "verbose", &nextword)) {
            log_init(LOG_VERB);
		} else if (is_word(line, "verify ", &nextword)) {
			sl_append(bp->verify_wcsfiles, nextword);
		} else if (is_word(line, "verify_wcs ", &nextword)) {
			sip_t wcs;
            int nread;
			memset(&wcs, 0, sizeof(wcs));
			if (sscanf(nextword, "%lg %lg %lg %lg %lg %lg %lg %lg%n",
			           &(wcs.wcstan.crval[0]), &(wcs.wcstan.crval[1]),
			           &(wcs.wcstan.crpix[0]), &(wcs.wcstan.crpix[1]),
			           &(wcs.wcstan.cd[0][0]), &(wcs.wcstan.cd[0][1]),
			           &(wcs.wcstan.cd[1][0]), &(wcs.wcstan.cd[1][1]),
                       &nread) < 8) {
				logerr("Failed to parse verify_wcs entry.\n");
				continue;
			}
            nextword += nread;
            if (sscanf(nextword, " %i %i%n", &(wcs.a_order), &(wcs.ap_order), &nread) >= 2) {
                int m, n;
                wcs.b_order = wcs.a_order;
                wcs.bp_order = wcs.ap_order;
                nextword += nread;
                for (m=0; m<=wcs.a_order; m++) {
                    for (n=0; (m+n)<=wcs.a_order; n++) {
                        if (sscanf(nextword, " %lg %lg%n", &(wcs.a[m][n]), &(wcs.b[m][n]), &nread) < 2) {
                            logerr("Failed to parse SIP terms: \"%s\"\n", nextword);
                            continue;
                        }
                        nextword += nread;
                    }
                }
                for (m=0; m<=wcs.ap_order; m++) {
                    for (n=0; (m+n)<=wcs.ap_order; n++) {
                        if (sscanf(nextword, " %lg %lg%n", &(wcs.ap[m][n]), &(wcs.bp[m][n]), &nread) < 2) {
                            logerr("Failed to parse SIP terms: \"%s\"\n", nextword);
                            continue;
                        }
                        nextword += nread;
                    }
                }
            }
            blind_add_verify_wcs(bp, &wcs);
		} else if (is_word(line, "cpulimit ", &nextword)) {
			bp->cpulimit = atof(nextword);
		} else if (is_word(line, "timelimit ", &nextword)) {
			bp->timelimit = atoi(nextword);
		} else if (is_word(line, "total_timelimit ", &nextword)) {
			bp->total_timelimit = atof(nextword);
		} else if (is_word(line, "total_cpulimit ", &nextword)) {
			bp->total_cpulimit = atof(nextword);
		} else if (is_word(line, "verify_pix ", &nextword)) {
			sp->verify_pix = atof(nextword);
		} else if (is_word(line, "nsolves ", &nextword)) {
			bp->nsolves = atoi(nextword);
		} else if (is_word(line, "ratio_tosolve ", &nextword)) {
			bp->logratio_tosolve = log(atof(nextword));
		} else if (is_word(line, "ratio_tokeep ", &nextword)) {
			sp->logratio_tokeep = log(atof(nextword));
		} else if (is_word(line, "ratio_toprint ", &nextword)) {
			sp->logratio_toprint = log(atof(nextword));
		} else if (is_word(line, "ratio_tobail ", &nextword)) {
			sp->logratio_bail_threshold = log(atof(nextword));
		} else if (is_word(line, "match ", &nextword)) {
            blind_set_match_file(bp, nextword);
		} else if (is_word(line, "indexrdls ", &nextword)) {
            blind_set_rdls_file(bp, nextword);
		} else if (is_word(line, "correspondences ", &nextword)) {
			free(bp->corr_fname);
			bp->corr_fname = strdup(nextword);
		} else if (is_word(line, "best_only", &nextword)) {
			bp->best_hit_only = TRUE;
		} else if (is_word(line, "solved ", &nextword)) {
            blind_set_solved_file(bp, nextword);
		} else if (is_word(line, "solved_in ", &nextword)) {
            blind_set_solvedin_file(bp, nextword);
		} else if (is_word(line, "solved_out ", &nextword)) {
            blind_set_solvedout_file(bp, nextword);
		} else if (is_word(line, "cancel ", &nextword)) {
            blind_set_cancel_file(bp, nextword);
		} else if (is_word(line, "solvedserver ", &nextword)) {
			free(bp->solvedserver);
			bp->solvedserver = strdup(nextword);
		} else if (is_word(line, "tweak_skipshift", &nextword)) {
			// deprecated
		} else if (is_word(line, "tweak_aborder ", &nextword)) {
			sp->tweak_aborder = atoi(nextword);
		} else if (is_word(line, "tweak_abporder ", &nextword)) {
			sp->tweak_abporder = atoi(nextword);
		} else if (is_word(line, "tweak", &nextword)) {
			sp->do_tweak = TRUE;
		} else if (is_word(line, "wcs ", &nextword)) {
            blind_set_wcs_file(bp, nextword);
		} else if (is_word(line, "fieldid_key ", &nextword)) {
			free(bp->fieldid_key);
			bp->fieldid_key = strdup(nextword);
		} else if (is_word(line, "maxquads ", &nextword)) {
			sp->maxquads = atoi(nextword);
		} else if (is_word(line, "maxmatches ", &nextword)) {
			sp->maxmatches = atoi(nextword);
		} else if (is_word(line, "xcol ", &nextword)) {
            blind_set_xcol(bp, nextword);
		} else if (is_word(line, "ycol ", &nextword)) {
            blind_set_ycol(bp, nextword);
		} else if (is_word(line, "index ", &nextword)) {
            blind_add_index(bp, nextword);
		} else if (is_word(line, "indexes_inparallel", &nextword)) {
			bp->indexes_inparallel = TRUE;
		} else if (is_word(line, "field ", &nextword)) {
            blind_set_field_file(bp, nextword);
		} else if (is_word(line, "fieldw ", &nextword)) {
			sp->field_maxx = atof(nextword);
		} else if (is_word(line, "fieldh ", &nextword)) {
			sp->field_maxy = atof(nextword);
		} else if (is_word(line, "distractors ", &nextword)) {
			sp->distractor_ratio = atof(nextword);
		} else if (is_word(line, "fieldid ", &nextword)) {
			bp->fieldid = atoi(nextword);
		} else if (is_word(line, "sdepth ", &nextword)) {
			sp->startobj = atoi(nextword);
		} else if (is_word(line, "depth ", &nextword)) {
			sp->endobj = atoi(nextword);
		} else if (is_word(line, "tol ", &nextword)) {
			sp->codetol = atof(nextword);
		} else if (is_word(line, "parity ", &nextword)) {
			sp->parity = atoi(nextword);
		} else if (is_word(line, "quadsize_min ", &nextword)) {
			sp->quadsize_min = atof(nextword);
		} else if (is_word(line, "fieldunits_lower ", &nextword)) {
			sp->funits_lower = atof(nextword);
		} else if (is_word(line, "fieldunits_upper ", &nextword)) {
			sp->funits_upper = atof(nextword);
		} else if (is_word(line, "firstfield ", &nextword)) {
			bp->firstfield = atoi(nextword);
		} else if (is_word(line, "lastfield ", &nextword)) {
			bp->lastfield = atoi(nextword);
		} else if (is_word(line, "fields ", &nextword)) {
			char* str = nextword;
            while (str && *str) {
                unsigned int lo, hi;
                int nread;
                if (sscanf(str, "%u%*1[-/]%u", &lo, &hi) == 2) {
                    sscanf(str, "%*u%*1[-/]%*u%n", &nread);
                } else if (sscanf(str, "%u", &lo) == 1) {
                    sscanf(str, "%*u%n", &nread);
                    hi = lo;
                } else {
                    fprintf(stderr, "Failed to parse fields fragment: \"%s\"\n", str);
                    return -1;
                }
                if (lo < 1) {
                    fprintf(stderr, "Field number %i is invalid: must be >= 1.\n", lo);
                    return -1;
                }
                if (lo > hi) {
                    fprintf(stderr, "Field range %i to %i is invalid: max must be >= min!\n", lo, hi);
                    return -1;
                }
                blind_add_field_range(bp, lo, hi);
                str += nread;
                while ((*str == ',') || isspace(*str))
                    str++;
            }
		} else if (is_word(line, "run", &nextword)) {
			return 0;
		} else if (is_word(line, "quit", &nextword)) {
			return 1;

		} else if (is_word(line, "nverify ", &nextword)) {
			logmsg("DEPRECATED: \"nverify\" command.\n");
		} else if (is_word(line, "nindex_tosolve ", &nextword)) {
			logmsg("DEPRECATED: \"nindex_tosolve\" command.\n");
		} else if (is_word(line, "nindex_tokeep ", &nextword)) {
			logmsg("DEPRECATED: \"nindex_tokeep\" command.\n");

		} else {
			logmsg("I didn't understand this command:\n  \"%s\"\n", line);
		}
	}
}
