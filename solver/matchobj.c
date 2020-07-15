/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include "os-features.h"
#include "sip.h"
#include "matchobj.h"
#include "starutil.h"
#include "log.h"
#include "verify.h"

char* matchobj_hit_miss_string(int* theta, int* testperm, int nbest,
                               int nfield, char* target) {
    int i;
    char* cur;
    if (!target) {
        target = malloc(256);
    }
    cur = target;
    for (i=0; i<MIN(nfield, 100); i++) {
        int ti = (testperm ? theta[testperm[i]] : theta[i]);
        if (ti == THETA_DISTRACTOR) {
            //loglevel(loglev, "-");
            *cur = '-';
            cur++;
        } else if (ti == THETA_CONFLICT) {
            //loglevel(loglev, "c");
            *cur = 'c';
            cur++;
        } else if (ti == THETA_FILTERED) {
            //loglevel(loglev, "f");
            *cur = 'f';
            cur++;
        } else if (ti == THETA_BAILEDOUT) {
            //loglevel(loglev, " bail");
            strcpy(cur, " bail");
            cur += 5;
            break;
        } else if (ti == THETA_STOPPEDLOOKING) {
            //loglevel(loglev, " stopped");
            strcpy(cur, " stopped");
            cur += 8;
            break;
        } else {
            //loglevel(loglev, "+");
            *cur = '+';
            cur++;
        }
        if (i+1 == nbest) {
            //loglevel(loglev, "(best)");
            strcpy(cur, "(best)");
            cur += 6;
        }
    }
    *cur = '\n';
    cur++;
    *cur = '\0';
    return target;
}

void matchobj_log_hit_miss(int* theta, int* testperm, int nbest, int nfield, int loglev, const char* prefix) {
    int n = strlen(prefix);
    char* buf = malloc(120 + n);
    strcpy(buf, prefix);
    matchobj_hit_miss_string(theta, testperm, nbest, nfield, buf + n);
    loglevel(loglev, "%s", buf);
    free(buf);
}


void matchobj_print(MatchObj* mo, int loglvl) {
    double ra,dec;
    loglevel(loglvl, "  log-odds ratio %g (%g), %i match, %i conflict, %i distractors, %i index.\n",
             mo->logodds, exp(mo->logodds), mo->nmatch, mo->nconflict, mo->ndistractor, mo->nindex);
    xyzarr2radecdeg(mo->center, &ra, &dec);
    loglevel(loglvl, "  RA,Dec = (%g,%g), pixel scale %g arcsec/pix.\n",
             ra, dec, mo->scale);
    if (mo->theta && mo->testperm) {
        loglevel(loglvl, "  Hit/miss: ");
        matchobj_log_hit_miss(mo->theta, mo->testperm, mo->nbest, mo->nfield,
                              loglvl, "  Hit/miss: ");
    }
}

void matchobj_compute_derived(MatchObj* mo) {
    int mx;
    int i;
    mx = 0;
    for (i=0; i<mo->dimquads; i++)
        mx = MAX(mx, mo->field[i]);
    mo->objs_tried = mx+1;
    if (mo->wcs_valid)
        mo->scale = tan_pixel_scale(&(mo->wcstan));
    mo->radius = deg2dist(mo->radius_deg);
    mo->nbest = mo->nmatch + mo->ndistractor + mo->nconflict;
}

/*void matchobj_log_verify_hit_miss(MatchObj* mo, int loglevel) {
 verify_log_hit_miss(mo->theta, mo->nbest, mo->nfield, loglevel);
 }
 */

const char* matchobj_get_index_name(MatchObj* mo) {
    if (!mo->index)
        return NULL;
    return mo->index->indexname;
}
