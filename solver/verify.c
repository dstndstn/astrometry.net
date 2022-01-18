/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "os-features.h"
#include "verify.h"
#include "permutedsort.h"
#include "mathutil.h"
#include "keywords.h"
#include "log.h"
#include "sip-utils.h"
#include "healpix.h"
#include "datalog.h"

#define DEBUGVERIFY 0

#if DEBUGVERIFY
#define debug2(args...) logdebug(args)
#else
#define debug2(args...)
#endif

#define DATALOG_MASK_VERIFY 0x1

// level
#define DLOG_ODDS 10

#define DLOG_ODDS_MIN log(1e6)

#define dlog(lev, fmt, ...) data_log(DATALOG_MASK_VERIFY, lev, fmt, ##__VA_ARGS__)

struct verify_s {
    const sip_t* wcs;

    // Reference stars:
    int NR;
    int NRall;
    int* refperm;
    int* refstarid;
    double* refxy;
    // temp storage when filtering
    int* badguys;

    // Image stars:
    int NT;
    int NTall;
    int* testperm;
    double* testxy;
    double* testsigma; // actually sigma**2.
    // temp storage
    int* tbadguys;

};
typedef struct verify_s verify_t;

static anbool* verify_deduplicate_field_stars(verify_t* v, const verify_field_t* vf, double nsigmas);

verify_field_t* verify_field_preprocess(const starxy_t* fieldxy) {
    verify_field_t* vf;
    int Nleaf = 5;

    vf = malloc(sizeof(verify_field_t));
    if (!vf) {
        fprintf(stderr, "Failed to allocate space for a verify_field_t().\n");
        return NULL;
    }
    vf->field = fieldxy;
    // Note on kdtree type: I tried U32 (duu) but it was marginally slower.
    // I didn't try U16 (dss) because we need a fair bit of accuracy here.
    // Make a copy of the field objects, because we're going to build a
    // kdtree out of them and that shuffles their order.
    vf->fieldcopy = starxy_copy_xy(fieldxy);
    vf->xy = starxy_copy_xy(fieldxy);
    if (!vf->fieldcopy || !vf->xy) {
        fprintf(stderr, "Failed to copy the field.\n");
        return NULL;
    }
    // Build a tree out of the field objects (in pixel space)
    vf->ftree = kdtree_build(NULL, vf->fieldcopy, starxy_n(vf->field),
                             2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);

    vf->do_uniformize = TRUE;
    vf->do_dedup = TRUE;
    vf->do_ror = TRUE;

    return vf;
}

void verify_field_free(verify_field_t* vf) {
    if (!vf)
        return;
    kdtree_free(vf->ftree);
    free(vf->xy);
    free(vf->fieldcopy);
    free(vf);
}

static double get_sigma2_at_radius(double verify_pix2, double r2, double quadr2) {
    return verify_pix2 * (1.0 + r2/quadr2);
}

static double* compute_sigma2s(const verify_field_t* vf,
                               const double* xy, int NF,
                               const double* qc, double Q2,
                               double verify_pix2, anbool do_gamma) {
    double* sigma2s;
    int i;
    double R2;

    sigma2s = malloc(NF * sizeof(double));
    if (!do_gamma) {
        for (i=0; i<NF; i++)
            sigma2s[i] = verify_pix2;
    } else {
        // Compute individual positional variances for every field
        // star.
        for (i=0; i<NF; i++) {
            if (vf) {
                double sxy[2];
                starxy_get(vf->field, i, sxy);
                // Distance from the quad center of this field star:
                R2 = distsq(sxy, qc, 2);
            } else
                R2 = distsq(xy + 2*i, qc, 2);

            // Variance of a field star at that distance from the quad center:
            sigma2s[i] = get_sigma2_at_radius(verify_pix2, R2, Q2);
        }
    }
    return sigma2s;
}

double* verify_compute_sigma2s(const verify_field_t* vf, const MatchObj* mo,
                               double verify_pix2, anbool do_gamma) {
    int NF;
    double qc[2];
    double Q2=0;
    NF = starxy_n(vf->field);
    if (do_gamma) {
        verify_get_quad_center(vf, mo, qc, &Q2);
        debug2("Quad radius = %g pixels\n", sqrt(Q2));
    }
    return compute_sigma2s(vf, NULL, NF, qc, Q2, verify_pix2, do_gamma);
}

double* verify_compute_sigma2s_arr(const double* xy, int NF,
                                   const double* qc, double Q2,
                                   double verify_pix2, anbool do_gamma) {
    return compute_sigma2s(NULL, xy, NF, qc, Q2, verify_pix2, do_gamma);
}

static double logd_at(double distractor, int mu, int NR, double logbg) {
    return log(distractor + (1.0-distractor)*mu / (double)NR) + logbg;
}

static int get_xy_bin(const double* xy,
                      double fieldW, double fieldH,
                      int nw, int nh) {
    int ix, iy;
    ix = (int)floor(nw * xy[0] / fieldW);
    ix = MAX(0, MIN(nw-1, ix));
    iy = (int)floor(nh * xy[1] / fieldH);
    iy = MAX(0, MIN(nh-1, iy));
    return iy * nw + ix;
}

static void print_test_perm(verify_t* v) {
    int i;
    for (i=0; i<v->NTall; i++) {
        if (i == v->NT)
            debug2("(NT)");
        debug2("%i ", v->testperm[i]);
    }
}

static void verify_get_test_stars(verify_t* v, const verify_field_t* vf, MatchObj* mo,
                                  double pix2, anbool do_gamma, anbool fake_match) {
    anbool* keepers = NULL;
    int i;
    int ibad=0, igood=0;

    v->NTall = starxy_n(vf->field);
    v->testxy = vf->xy;
    v->NT = v->NTall;
    v->testsigma = verify_compute_sigma2s(vf, mo, pix2, do_gamma);
    v->testperm = permutation_init(NULL, v->NTall);
    v->tbadguys = malloc(v->NTall * sizeof(int));

    if (DEBUGVERIFY) {
        debug2("start:\n");
        print_test_perm(v);
        debug2("\n");
    }

    if (vf->do_dedup) {
        // Deduplicate test stars.  This could be done (approximately) in preprocessing.
        // FIXME -- this should be at the reference deduplication radius, not relative to sigma!
        // -- this requires the match scale
        // -- can perhaps discretize dedup to nearest power-of-sqrt(2) pixel radius and cache it.
        // -- we can compute sigma much later
        keepers = verify_deduplicate_field_stars(v, vf, 1.0);

        // Remove test quad stars.  Do this after deduplication so we
        // don't end up with (duplicate) test stars near the quad stars.
        if (!fake_match) {
            for (i=0; i<mo->dimquads; i++) {
                assert(mo->field[i] >= 0);
                assert(mo->field[i] < v->NTall);
                keepers[mo->field[i]] = FALSE;
            }
        }

        ibad = igood = 0;
        for (i=0; i<v->NT; i++) {
            int ti = v->testperm[i];
            if (keepers[ti]) {
                v->testperm[igood] = ti;
                igood++;
            } else {
                v->tbadguys[ibad] = ti;
                ibad++;
            }
        }
    } else {
        // Remove the quad.
        if (!fake_match) {
            int j;
            for (i=0; i<mo->dimquads; i++) {
                assert(mo->field[i] >= 0);
                assert(mo->field[i] < v->NTall);
            }
            ibad = igood = 0;
            for (i=0; i<v->NT; i++) {
                int ti = v->testperm[i];
                anbool isquad = FALSE;
                for (j=0; j<mo->dimquads; j++) {
                    if (ti == mo->field[j]) {
                        isquad = TRUE;
                        break;
                    }
                }
                if (!isquad) {
                    v->testperm[igood] = ti;
                    igood++;
                } else {
                    v->tbadguys[ibad] = ti;
                    ibad++;
                }
            }
        } else {
            igood = v->NT;
        }
    }

    v->NT = igood;
    // remember the bad guys
    memcpy(v->testperm + igood, v->tbadguys, ibad * sizeof(int));
    free(keepers);

    if (DEBUGVERIFY) {
        debug2("after dedup and removing quad:\n");
        print_test_perm(v);
        debug2("\n");
    }

}

double verify_get_ror2(double Q2, double area,
                       double distractors, int NR, double pix2) {
    return Q2 * MAX(1, (area*(1 - distractors) / (4. * M_PI * NR * pix2) - 1));
}

static void verify_apply_ror(verify_t* v,
                             int index_cutnside,
                             MatchObj* mo,
                             const verify_field_t* vf,
                             double pix2,
                             double distractors,
                             double fieldW,
                             double fieldH,
                             anbool do_gamma, anbool fake_match,
                             double* p_effA,
                             int* p_uninw, int* p_uninh) {
    int i;
    int uni_nw = 0, uni_nh = 0;
    double effA = fieldW * fieldH;
    double qc[2], Q2=0;
    int igood, ibad;
    int* binids = NULL;
    double* bincenters = NULL;

    // If we're verifying an existing WCS solution, then don't increase the variance
    // away from the center of the matched quad.
    if (fake_match)
        do_gamma = FALSE;

    verify_get_test_stars(v, vf, mo, pix2, do_gamma, fake_match);
    debug2("Number of test stars: %i\n", v->NT);
    debug2("Number of reference stars: %i\n", v->NR);

    if (!fake_match)
        verify_get_quad_center(vf, mo, qc, &Q2);

    // Uniformize test stars
    // FIXME - can do this (possibly at several scales) in preprocessing.
    if (vf->do_uniformize) {
        // -get uniformization scale.
        verify_get_uniformize_scale(index_cutnside, mo->scale, fieldW, fieldH, &uni_nw, &uni_nh);
        debug2("uniformizing into %i x %i blocks.\n", uni_nw, uni_nh);

        // uniformize!
        if (uni_nw > 1 || uni_nh > 1) {
            verify_uniformize_field(vf->xy, v->testperm, v->NT, fieldW, fieldH, uni_nw, uni_nh, NULL, &binids);
            bincenters = verify_uniformize_bin_centers(fieldW, fieldH, uni_nw, uni_nh);

            if (DEBUGVERIFY) {
                debug2("after uniformizing:\n");
                print_test_perm(v);
                debug2("\n");
            }
        }
    }
    if (vf->do_ror && !fake_match) {
        anbool* goodbins = NULL;
        int Ngoodbins;
        double ror2;

        debug2("Quad radius = %g\n", sqrt(Q2));
        ror2 = verify_get_ror2(Q2, fieldW*fieldH, distractors, v->NR, pix2);
        debug2("(strong) Radius of relevance is %.1f\n", sqrt(ror2));

        if (binids) {
            assert(uni_nw);
            goodbins = malloc((size_t)uni_nw * (size_t)uni_nh * sizeof(anbool));
            Ngoodbins = 0;
            for (i=0; i<(uni_nw * uni_nh); i++) {
                double binr2 = distsq(bincenters + 2*i, qc, 2);
                goodbins[i] = (binr2 < ror2);
                if (goodbins[i])
                    Ngoodbins++;
            }
            // Remove test stars in irrelevant bins...
            igood = ibad = 0;
            for (i=0; i<v->NT; i++) {
                int ti = v->testperm[i];
                if (goodbins[binids[i]]) {
                    v->testperm[igood] = ti;
                    igood++;
                } else {
                    v->tbadguys[ibad] = ti;
                    ibad++;
                }
            }
        } else {
            // Remove test stars outside the RoR.
            igood = ibad = 0;
            for (i=0; i<v->NT; i++) {
                int ti = v->testperm[i];
                double r2 = distsq(qc, vf->xy + 2*ti, 2);
                if (r2 < ror2) {
                    v->testperm[igood] = ti;
                    igood++;
                } else {
                    v->tbadguys[ibad] = ti;
                    ibad++;
                }
            }
            // Count good bins to find effective area... (ugh)
            assert(!bincenters);
            if (!uni_nw)
                verify_get_uniformize_scale(index_cutnside, mo->scale, fieldW, fieldH, &uni_nw, &uni_nh);
            bincenters = verify_uniformize_bin_centers(fieldW, fieldH, uni_nw, uni_nh);
            Ngoodbins = 0;
            for (i=0; i<(uni_nw * uni_nh); i++) {
                double binr2 = distsq(bincenters + 2*i, qc, 2);
                if (binr2 < ror2)
                    Ngoodbins++;
            }
        }

        v->NT = igood;
        memcpy(v->testperm + igood, v->tbadguys, ibad * sizeof(int));
        debug2("After removing %i/%i irrelevant bins: %i test stars.\n", (uni_nw*uni_nh)-Ngoodbins, uni_nw*uni_nh, v->NT);

        if (DEBUGVERIFY) {
            debug2("after applying RoR:\n");
            print_test_perm(v);
            debug2("\n");
        }

        // Effective area: A * proportion of good bins.
        effA *= Ngoodbins / (double)(uni_nw * uni_nh);

        // Remove reference stars in bad bins.
        igood = ibad = 0;
        if (goodbins) {
            assert(uni_nw);
            for (i=0; i<v->NR; i++) {
                int ri = v->refperm[i];
                int binid = get_xy_bin(v->refxy + 2*ri, fieldW, fieldH, uni_nw, uni_nh);
                if (goodbins[binid]) {
                    v->refperm[igood] = ri;
                    igood++;
                } else {
                    v->badguys[ibad] = ri;
                    ibad++;
                }
            }
        } else {
            for (i=0; i<v->NR; i++) {
                int ri = v->refperm[i];
                if (distsq(qc, v->refxy + 2*ri, 2) < ror2) {
                    v->refperm[igood] = ri;
                    igood++;
                } else {
                    v->badguys[ibad] = ri;
                    ibad++;
                }
            }
        }
        // remember the bad guys
        memcpy(v->refperm + igood, v->badguys, ibad * sizeof(int));
        v->NR = igood;
        debug2("After removing irrelevant ref stars: %i ref stars.\n", v->NR);

        // New ROR is...
        debug2("ROR changed from %g to %g\n", sqrt(ror2),
               sqrt(verify_get_ror2(Q2, effA, distractors, v->NR, pix2)));

        free(goodbins);
    }
    free(bincenters);
    free(binids);

    *p_effA = effA;
    if (p_uninw)
        *p_uninw = uni_nw;
    if (p_uninh)
        *p_uninh = uni_nh;
}

static double real_verify_star_lists(verify_t* v,
                                     double effective_area,
                                     double distractors,
                                     double logodds_bail,
                                     double logodds_stoplooking,
                                     int* p_besti,
                                     double** p_logodds, int** p_theta,
                                     double* p_worstlogodds,
                                     int* p_ibailed, int* p_istopped) {
    int i, j;
    double worstlogodds;
    double bestworstlogodds;
    double bestlogodds;
    int besti;
    double logodds;
    double logbg;
    double logd;
    //double matchnsigma = 5.0;
    double* refcopy;
    kdtree_t* rtree;
    int Nleaf = 10;
    int* rmatches;
    double* rprobs;
    double* all_logodds = NULL;
    int* theta = NULL;
    int mu;
    int* rperm;

    if (!v->NR || !v->NT) {
        logerr("real_verify_star_lists: NR=%i, NT=%i\n", v->NR, v->NT);
        return -LARGE_VAL;
    }

    // Build a tree out of the index stars in pixel space...
    // kdtree scrambles the data array so make a copy first.
    refcopy = malloc(2 * v->NR * sizeof(double));
    // we must pack/unpermute the refxys; remember this packing order in "rperm".
    // we borrow storage for "rperm"...
    if (!v->badguys)
        v->badguys = malloc(v->NR * sizeof(int));
    rperm = v->badguys;
    for (i=0; i<v->NR; i++) {
        int ri = v->refperm[i];
        rperm[i] = ri;
        refcopy[2*i+0] = v->refxy[2*ri+0];
        refcopy[2*i+1] = v->refxy[2*ri+1];
    }
    rtree = kdtree_build(NULL, refcopy, v->NR, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);

    rmatches = malloc(v->NR * sizeof(int));
    for (i=0; i<v->NR; i++)
        rmatches[i] = -1;

    rprobs = malloc(v->NR * sizeof(double));
    for (i=0; i<v->NR; i++)
        rprobs[i] = -LARGE_VAL;

    if (p_logodds || data_log_passes(DATALOG_MASK_VERIFY, DLOG_ODDS))
        all_logodds = calloc(v->NT, sizeof(double));
    if (p_logodds)
        *p_logodds = all_logodds;
	
    if (p_ibailed)
        *p_ibailed = -1;
    if (p_istopped)
        *p_istopped = -1;

    theta = malloc(v->NT * sizeof(int));

    logbg = log(1.0 / effective_area);

    worstlogodds = 0;
    bestlogodds = -LARGE_VAL;
    bestworstlogodds = -LARGE_VAL;
    besti = -1;
    logodds = 0.0;
    mu = 0;
    for (i=0; i<v->NT; i++) {
        const double* testxy;
        double sig2;
        int refi;
        int tmpi;
        double d2;
        //double reallogfg;
        double logfg;
        int ti;

        ti = v->testperm[i];
        testxy = v->testxy + 2*ti;
        sig2 = v->testsigma[ti];

        logd = logd_at(distractors, mu, v->NR, logbg);

        debug2("\n");
        debug2("test star %i: (%.1f,%.1f), sigma: %.1f\n", i, testxy[0], testxy[1], sqrt(sig2));

        // find nearest ref star (within 5 sigma)
        tmpi = kdtree_nearest_neighbour_within(rtree, testxy, sig2 * 25.0, &d2);
        if (tmpi == -1) {
            // no nearest neighbour within range.
            debug2("  No nearest neighbour.\n");
            refi = -1;
            logfg = -LARGE_VAL;
        } else {
            double loggmax;
            // Note that "refi" is w.r.t. the "refcopy" array (not the original data).
            refi = kdtree_permute(rtree, tmpi);
            // peak value of the Gaussian
            loggmax = log((1.0 - distractors) / (2.0 * M_PI * sig2 * v->NR));
            // FIXME - do something with uninformative hits?
            // these should be eliminated by RoR filtering...
            if (loggmax < logbg)
                debug2("  This star is uninformative: peak %.1f, bg %.1f.\n", loggmax, logbg);

            // value of the foreground Gaussian
            logfg = loggmax - d2 / (2.0 * sig2);
			
            debug2("  NN: ref star %i, dist %.2f, sigmas: %.3f, logfg: %.1f (%.1f above distractor, %.1f above bg)\n",
                   refi, sqrt(d2), sqrt(d2 / sig2), logfg, logfg - logd, logfg - logbg);
        }

        if (logfg < logd) {
            //reallogfg = 
            logfg = logd;
            debug2("  Distractor.\n");
            theta[i] = THETA_DISTRACTOR;
        } else {
            // duplicate match?
            if (rmatches[refi] != -1) {
                double oldfg = rprobs[refi];
                //debug2("Conflict: odds was %g, now %g.\n", oldfg, logfg);
                // Conflict.  Compute probabilities of old vs new theta.
                // if we keep the old one: the new star is a distractor
                double keepfg = logd;

                // if we switch to the new one: the new star is a match...
                double switchfg = logfg;
                // ... and the old one becomes a distractor...
                int oldj = rmatches[refi];
                int muj = 0;
                //reallogfg = logfg;
                for (j=0; j<oldj; j++)
                    if (theta[j] >= 0)
                        muj++;
                switchfg += (logd_at(distractors, muj, v->NR, logbg) - oldfg);
                // FIXME - could estimate/bound the distractor change and avoid computing it...

                // ... and the intervening distractors become worse.
                debug2("  oldj is %i, muj is %i.\n", oldj, muj);
                debug2("  changing old point to distractor: %.1f change in logodds\n",
                       (logd_at(distractors, muj, v->NR, logbg) - oldfg));
                for (; j<i; j++)
                    if (theta[j] < 0) {
                        switchfg += (logd_at(distractors, muj, v->NR, logbg) -
                                     logd_at(distractors, muj+1, v->NR, logbg));
                        debug2("  adjusting distractor %i: %g change in logodds\n",
                               j, (logd_at(distractors, muj, v->NR, logbg) -
                                   logd_at(distractors, muj+1, v->NR, logbg)));
                    } else
                        muj++;
                debug2("  Conflict: keeping   old match, logfg would be %.1f\n", keepfg);
                debug2("  Conflict: accepting new match, logfg would be %.1f\n", switchfg);
				
                if (switchfg > keepfg) {
                    // upgrade: old match becomes a distractor.
                    debug2("  Conflict: upgrading.\n");
                    theta[oldj] = THETA_CONFLICT;
                    // Note that here we want the entries in "theta" to be
                    // indices into "v->refxy" et al, so apply the "rperm" permutation.
                    theta[i] = rperm[refi];
                    // record this new match.
                    rmatches[refi] = i;
                    rprobs[refi] = logfg;

                    // "switchfg" incorporates the cost of adjusting the previous probabilities.
                    logfg = switchfg;

                    // FIXME -- Do we need to repeat the distractor-adjustment
                    // loop above, updating all_logodds entries??
                    // No, not really -- we update "logfg" in this loop, and record it below
                    // and that's sort of right -- it's THIS star that resulting in all the changes.
                    /*
                     if (all_logodds) {
                     muj = 0;
                     for (j=0; j<oldj; j++)
                     if (theta[j] >= 0)
                     muj++;
                     all_logodds[oldj] = logd_at(distractors, muj, v->NR, logbg) - logbg;
                     for (j=oldj; j<i; j++)
                     if (theta[j] < 0) {
                     all_logodds[j] = logd_at(distractors, muj, v->NR, logbg) - logbg;
                     } else {
                     muj++;
                     }
                     double logp = 0.;
                     for (j=0; j<i; j++)
                     logp += all_logodds[j];
                     logverb("updated all_logodds = %g, vs logodds %g\n",
                     logp, logodds);
                     }
                     */


                } else {
                    // old match was better: this match becomes a distractor.
                    debug2("  Conflict: not upgrading.\n"); //  logprob was %.1f, now %.1f.\n", oldfg, logfg);
                    logfg = keepfg;
                    theta[i] = THETA_CONFLICT;
                }
                // no change in mu.

            } else {
                // new match.
                rmatches[refi] = i;
                rprobs[refi] = logfg;
                theta[i] = rperm[refi];
                mu++;
            }
        }

        logodds += (logfg - logbg);
        debug2("  Logodds: change %.1f, now %.1f\n", (logfg - logbg), logodds);

        if (all_logodds)
            all_logodds[i] = logfg - logbg;

        if (logodds < logodds_bail) {
            debug2("  logodds %g less than bailout %g\n", logodds, logodds_bail);
            if (p_ibailed)
                *p_ibailed = i;
            break;
        }

        worstlogodds = MIN(worstlogodds, logodds);

        if (logodds > bestlogodds) {
            bestlogodds = logodds;
            besti = i;
            // Record the worst log-odds we've seen up to this point.
            bestworstlogodds = worstlogodds;
        }

        if (logodds > logodds_stoplooking) {
            if (p_istopped)
                *p_istopped = i;
            break;
        }
    }

    if (bestlogodds > DLOG_ODDS_MIN) {
        // when the loop stopped...
        int iend = i;
        data_log_start_item(DATALOG_MASK_VERIFY, DLOG_ODDS, "logodds");
        dlog(DLOG_ODDS, "[");
        for (i=0; i<iend; i++)
            dlog(DLOG_ODDS, "%s%g", (i ? ", ":""), all_logodds[i]);
        dlog(DLOG_ODDS, "]");
        data_log_end_item(DATALOG_MASK_VERIFY, DLOG_ODDS);

        data_log_start_item(DATALOG_MASK_VERIFY, DLOG_ODDS, "bestlogodds");
        dlog(DLOG_ODDS, "%g", bestlogodds);
        data_log_end_item(DATALOG_MASK_VERIFY, DLOG_ODDS);

        /*
         double lnp = 0.0;
         for (i=0; i<5; i++)
         lnp += all_logodds[i];
         if (lnp > 4.) {
         printf("lnp at step 5: %g\n", lnp);
         printf("test perm:");
         for (i=0; i<10; i++)
         printf(" %i", v->testperm[i]);
         printf("\n");
         printf("theta:");
         for (i=0; i<10; i++)
         printf(" %i", theta[i]);
         printf("\n");

         data_log_start_item(DATALOG_MASK_VERIFY, DLOG_ODDS, "match");
         dlog(DLOG_ODDS, "{ 'refxy': [");
         for (i=0; i<v->NRall; i++)
         dlog(DLOG_ODDS, "(%.3f,%.3f),", v->refxy[2*i+0], v->refxy[2*i+1]);
         dlog(DLOG_ODDS, "], 'testxy': [");
         for (i=0; i<v->NTall; i++)
         dlog(DLOG_ODDS, "(%.3f,%.3f),", v->testxy[2*i+0], v->testxy[2*i+1]);
         dlog(DLOG_ODDS, "], 'testperm': [");
         for (i=0; i<v->NT; i++)
         dlog(DLOG_ODDS, "%i,", v->testperm[i]);
         dlog(DLOG_ODDS, "], 'refperm': [");
         for (i=0; i<v->NR; i++)
         dlog(DLOG_ODDS, "%i,", v->refperm[i]);
         dlog(DLOG_ODDS, "], 'theta': [");
         for (i=0; i<v->NT; i++)
         dlog(DLOG_ODDS, "%i,", theta[i]);
         dlog(DLOG_ODDS, "], 'logodds5': %g, 'all_logodds': [", lnp);
         for (i=0; i<iend; i++)
         dlog(DLOG_ODDS, "%g,", all_logodds[i]);
         dlog(DLOG_ODDS, "] }");
         data_log_end_item(DATALOG_MASK_VERIFY, DLOG_ODDS);
         }
         */
    }

    free(rmatches);

    if (p_theta)
        *p_theta = theta;
    else
        free(theta);

    if (p_besti)
        *p_besti = besti;

    if (p_worstlogodds)
        *p_worstlogodds = bestworstlogodds;

    if (all_logodds && !*p_logodds)
        free(all_logodds);

    free(rprobs);

    kdtree_free(rtree);
    free(refcopy);

    return bestlogodds;
}

void verify_get_index_stars(const double* fieldcenter, double fieldr2,
                            const startree_t* skdt, const sip_t* sip, const tan_t* tan,
                            double fieldW, double fieldH,
                            double** p_indexradec,
                            double** indexpix, int** p_starids, int* p_nindex) {
    double* indxyz;
    int i, N, NI;
    int* sweep;
    int* starid;
    int* inbounds;
    int* perm;
    double* radec = NULL;

    assert(skdt->sweep);
    assert(p_nindex);
    assert(sip || tan);

    // Find all index stars within the bounding circle of the field.
    startree_search_for(skdt, fieldcenter, fieldr2, &indxyz, NULL, &starid, &N);

    if (!indxyz) {
        // no stars in range.
        *p_nindex = 0;
        return;
    }

    // Find index stars within the rectangular field.
    inbounds = sip_filter_stars_in_field(sip, tan, indxyz, NULL, N, indexpix,
                                         NULL, &NI);
    // Apply the permutation now, so that "indexpix" and "starid" stay in sync:
    // indexpix is already in the "inbounds" ordering.
    permutation_apply(inbounds, NI, starid, starid, sizeof(int));

    // Compute index RA,Decs if requested.
    if (p_indexradec) {
        radec = malloc(2 * NI * sizeof(double));
        for (i=0; i<NI; i++)
            // note that the "inbounds" permutation is applied to "indxyz" here.
            // we will apply the sweep permutation below.
            xyzarr2radecdegarr(indxyz + 3*inbounds[i], radec + 2*i);
        *p_indexradec = radec;
    }
    free(indxyz);
    free(inbounds);

    // Each index star has a "sweep number" assigned during index building;
    // it roughly represents a local brightness ordering.  Use this to sort the
    // index stars.
    sweep = malloc(NI * sizeof(int));
    for (i=0; i<NI; i++)
        sweep[i] = skdt->sweep[starid[i]];
    perm = permuted_sort(sweep, sizeof(int), compare_ints_asc, NULL, NI);
    free(sweep);

    if (indexpix) {
        permutation_apply(perm, NI, *indexpix, *indexpix, 2 * sizeof(double));
        *indexpix = realloc(*indexpix, NI * 2 * sizeof(double));
    }

    if (p_starids) {
        permutation_apply(perm, NI, starid, starid, sizeof(int));
        starid = realloc(starid, NI * sizeof(int));
        *p_starids = starid;
    } else
        free(starid);

    if (p_indexradec)
        permutation_apply(perm, NI, radec, radec, 2 * sizeof(double));

    free(perm);

    *p_nindex = NI;
}

/**
 If field objects are within "sigma" of each other (where sigma depends on the
 distance from the matched quad), then they are not very useful for verification.
 We filter out field stars within sigma of each other, taking only the brightest.

 Returns an array indicating which field stars should be kept.
 */
static anbool* verify_deduplicate_field_stars(verify_t* v, const verify_field_t* vf, double nsigmas) {
    anbool* keepers = NULL;
    int i, j, ti;
    kdtree_qres_t* res = NULL;
    double nsig2 = nsigmas*nsigmas;
    int options = KD_OPTIONS_NO_RESIZE_RESULTS | KD_OPTIONS_SMALL_RADIUS;

    // default to FALSE
    keepers = calloc(v->NTall, sizeof(anbool));
    for (i=0; i<v->NT; i++) {
        ti = v->testperm[i];
        keepers[ti] = TRUE;
    }
    for (i=0; i<v->NT; i++) {
        double sxy[2];
        ti = v->testperm[i];
        if (!keepers[ti])
            continue;
        starxy_get(vf->field, ti, sxy);
        res = kdtree_rangesearch_options_reuse(vf->ftree, res, sxy, nsig2 * v->testsigma[ti], options);
        for (j=0; j<res->nres; j++) {
            int ind = res->inds[j];
            if (ind > i) {
                keepers[ind] = FALSE;
                if (DEBUGVERIFY) {
                    double otherxy[2];
                    starxy_get(vf->field, ind, otherxy);
                    logdebug("Field star %i at %g,%g: is close to field star %i at %g,%g.  dist is %g, sigma is %g\n", 
                             i, sxy[0], sxy[1], ind, otherxy[0], otherxy[1],
                             sqrt(distsq(sxy, otherxy, 2)), sqrt(nsig2 * v->testsigma[ti]));
                }
            }
        }
    }
    kdtree_free_query(res);
    return keepers;
}

void verify_get_quad_center(const verify_field_t* vf, const MatchObj* mo, double* centerpix,
                            double* quadr2) {
    double Axy[2], Bxy[2];
    // Find the midpoint of AB of the quad in pixel space.
    starxy_get(vf->field, mo->field[0], Axy);
    starxy_get(vf->field, mo->field[1], Bxy);
    centerpix[0] = 0.5 * (Axy[0] + Bxy[0]);
    centerpix[1] = 0.5 * (Axy[1] + Bxy[1]);
    // Find the radius-squared of the quad = distsq(qc, A)
    *quadr2 = distsq(Axy, centerpix, 2);
}

void verify_get_uniformize_scale(int cutnside, double scale, int W, int H, int* cutnw, int* cutnh) {
    double cutarcsec, cutpix;
    cutarcsec = healpix_side_length_arcmin(cutnside) * 60.0;
    cutpix = cutarcsec / scale;
    debug2("cut nside: %i\n", cutnside);
    debug2("cut scale: %g arcsec\n", cutarcsec);
    debug2("match scale: %g arcsec/pix\n", scale);
    debug2("cut scale: %g pixels\n", cutpix);
    if (cutnw)
        *cutnw = MAX(1, (int)round(W / cutpix));
    if (cutnh)
        *cutnh = MAX(1, (int)round(H / cutpix));
}

void verify_uniformize_field(const double* xy,
                             int* perm,
                             int N,
                             double fieldW, double fieldH,
                             int nw, int nh,
                             int** p_bincounts,
                             int** p_binids) {
    il** lists;
    int i,j,k,p;
    int* bincounts = NULL;
    int* binids = NULL;

    if (p_binids) {
        binids = malloc((size_t)N * sizeof(int));
        *p_binids = binids;
    }

    lists = malloc((size_t)nw * (size_t)nh * sizeof(il*));
    for (i=0; i<(nw*nh); i++)
        lists[i] = il_new(16);

    // put the stars in the appropriate bins.
    debug2("Test star bins:\n");
    for (i=0; i<N; i++) {
        int ind;
        int bin;
        ind = perm[i];
        bin = get_xy_bin(xy + 2*ind, fieldW, fieldH, nw, nh);
        debug2("%i ", bin);
        il_append(lists[bin], ind);
    }
    debug2("\n");

    if (p_bincounts) {
        // note the bin occupancies.
        bincounts = malloc((size_t)nw * (size_t)nh * sizeof(int));
        for (i=0; i<(nw*nh); i++) {
            bincounts[i] = il_size(lists[i]);
            //logverb("bin %i has %i stars\n", i, bincounts[i]);
        }
        *p_bincounts = bincounts;
    }

    // make sweeps through the bins, grabbing one star from each.
    p=0;
    for (k=0;; k++) {
        for (j=0; j<nh; j++) {
            for (i=0; i<nw; i++) {
                int binid = j*nw + i;
                il* lst = lists[binid];
                if (k >= il_size(lst))
                    continue;
                perm[p] = il_get(lst, k);
                if (binids)
                    binids[p] = binid;
                p++;
            }
        }
        if (p == N)
            break;
    }
    assert(p == N);

    for (i=0; i<(nw*nh); i++)
        il_free(lists[i]);
    free(lists);
}

double* verify_uniformize_bin_centers(double fieldW, double fieldH,
                                      int nw, int nh) {
    int i,j;
    double* bxy = malloc((size_t)nw * (size_t)nh * (size_t)2 * sizeof(double));
    for (j=0; j<nh; j++)
        for (i=0; i<nw; i++) {
            bxy[(j * nw + i)*2 +0] = (i + 0.5) * fieldW / (double)nw;
            bxy[(j * nw + i)*2 +1] = (j + 0.5) * fieldH / (double)nh;
        }
    return bxy;
}

void verify_wcs(const startree_t* skdt,
                int index_cutnside,
                const sip_t* sip,
                const verify_field_t* vf,
                double verify_pix2,
                double distractors,
                double fieldW,
                double fieldH,
                double logbail,
                double logaccept,
                double logstoplooking,

                double* logodds,
                int* nfield, int* nindex,
                int* nmatch, int* nconflict, int* ndistractor
                // int** theta ?
                ) {
    MatchObj mo;

    memset(&mo, 0, sizeof(MatchObj));

    radecdeg2xyzarr(sip->wcstan.crval[0], sip->wcstan.crval[1], mo.center);
    mo.radius = arcsec2dist(hypot(fieldW, fieldH)/2.0 * sip_pixel_scale(sip));
    memcpy(&(mo.wcstan), &(sip->wcstan), sizeof(tan_t));
    mo.wcs_valid = TRUE;

    verify_hit(skdt, index_cutnside, &mo, sip, vf, verify_pix2,
               distractors, fieldW, fieldH, logbail, logaccept,
               logstoplooking, FALSE, TRUE);

    if (logodds)
        *logodds = mo.logodds;
    if (nfield)
        *nfield = mo.nfield;
    if (nindex)
        *nindex = mo.nindex;
    if (nmatch)
        *nmatch = mo.nmatch;
    if (nconflict)
        *nconflict = mo.nconflict;
    if (ndistractor)
        *ndistractor = mo.ndistractor;
}


static void set_null_mo(MatchObj* mo) {
    mo->nfield = 0;
    mo->nmatch = 0;
    matchobj_compute_derived(mo);
    mo->logodds = -LARGE_VAL;
}

static void check_permutation(const int* perm, int N) {
    int i;
    int* counts = calloc(N, sizeof(int));
    for (i=0; i<N; i++) {
        assert(perm[i] >= 0);
        assert(perm[i] < N);
        counts[perm[i]]++;
    }
    for (i=0; i<N; i++) {
        assert(counts[i] == 1);
    }
    free(counts);
}

static void fixup_theta(int* theta, double* allodds, int ibailed, int istopped, verify_t* v,
                        int besti, int NRimage, double* refxyz,
                        int** p_etheta, double** p_eodds) {
    int* etheta;
    double* eodds;
    int* invrperm;
    int i, ti;

    if (DEBUGVERIFY) {
        // The "testperm" permutation should be "complete".
        check_permutation(v->testperm, v->NTall);
        // "refperm" has vals < NRall in elements < NRimage.
        //check_permutation(v->refperm, NRimage);
        for (i=0; i<NRimage; i++) {
            assert(v->refperm[i] >= 0);
            assert(v->refperm[i] < v->NRall);
        }
    }

    // "theta" has length v->NT.

    if (ibailed != -1)
        for (i=ibailed+1; i<v->NT; i++)
            theta[i] = THETA_BAILEDOUT;

    if (istopped != -1)
        for (i=istopped+1; i<v->NT; i++)
            theta[i] = THETA_STOPPEDLOOKING;

    // At this point, "theta[0]" is the *reference* star index
    // that was matched by the test star "v->testperm[0]".
    // Meanwhile, "v->refperm" lists all the valid reference stars.

    // We want to produce "etheta", which has elements parallel to
    // the test stars in their original (brightness) ordering; that is,
    // we want to eliminate the need for "v->testperm".

    if (DEBUGVERIFY) {
        for (i=0; i<v->NT; i++) {
            Unused int ri;
            if (i == besti)
                debug2("* ");
            debug2("Theta[%i] = %i", i, theta[i]);
            if (theta[i] < 0) {
                debug2("\n");
                continue;
            }
            ri = theta[i];
            ti = v->testperm[i];
            debug2(" (starid %i), testxy=(%.1f, %.1f), refxy=(%.1f, %.1f)\n",
                   (v->refstarid ? v->refstarid[ri] : -1000), v->testxy[ti*2+0], v->testxy[ti*2+1], v->refxy[ri*2+0], v->refxy[ri*2+1]);
        }
    }
    etheta = malloc(v->NTall * sizeof(int));
    eodds = malloc(v->NTall * sizeof(double));

    // Apply the "refperm" permutation, mostly to cut out the stars that
    // aren't in the image (we want to have "nindex" = "NRimage" = "NRall").
    // This requires computing the inverse perm so we can fix theta to match.

    // The reference stars include stars that are actually outside
    // the field; we want to collapse the reference star list,
    // which will renumber them.

    invrperm = malloc(v->NRall * sizeof(int));
#define BAD_PERM -1000000
    if (DEBUGVERIFY) {
        for (i=0; i<v->NRall; i++)
            invrperm[i] = BAD_PERM;
    }
    for (i=0; i<NRimage; i++)
        invrperm[v->refperm[i]] = i;

    if (v->refstarid)
        permutation_apply(v->refperm, NRimage, v->refstarid, v->refstarid, sizeof(int));
    permutation_apply(v->refperm, NRimage, v->refxy, v->refxy, 2*sizeof(double));
    if (refxyz)
        permutation_apply(v->refperm, NRimage, refxyz, refxyz, 3*sizeof(double));

    // New v->refstarid[i] is old v->refstarid[ v->refperm[i] ]

    if (DEBUGVERIFY) {
        for (i=0; i<v->NTall; i++)
            etheta[i] = BAD_PERM;
    }

    for (i=0; i<v->NT; i++) {
        ti = v->testperm[i];
        if (DEBUGVERIFY)
            // assert that we haven't touched this element yet.
            assert(etheta[ti] == BAD_PERM);
        if (theta[i] < 0) {
            etheta[ti] = theta[i];
            // No match -> no weight.
            eodds[ti] = -LARGE_VAL;
        } else {
            if (DEBUGVERIFY)
                assert(invrperm[theta[i]] != BAD_PERM);
            etheta[ti] = invrperm[theta[i]];
            eodds[ti] = allodds[i];
        }
    }

    free(invrperm);

    for (i=v->NT; i<v->NTall; i++) {
        ti = v->testperm[i];
        etheta[ti] = THETA_FILTERED;
        eodds[ti] = -LARGE_VAL;
    }

    if (DEBUGVERIFY) {
        // We should touch every element.
        for (i=0; i<v->NTall; i++)
            assert(etheta[i] != BAD_PERM);
        for (i=0; i<v->NTall; i++)
            if (etheta[i] >= 0)
                assert(etheta[i] < NRimage);
            else
                assert(etheta[i] == THETA_FILTERED ||
                       etheta[i] == THETA_DISTRACTOR ||
                       etheta[i] == THETA_CONFLICT ||
                       etheta[i] == THETA_BAILEDOUT ||
                       etheta[i] == THETA_STOPPEDLOOKING);
					   
    }

    *p_etheta = etheta;
    *p_eodds = eodds;
}

void verify_count_hits(int* theta, int besti, int* p_nmatch, int* p_nconflict, int* p_ndistractor) {
    int i;
    int d, c, m;
    d = 0;
    c = 0;
    m = 0;
    for (i=0; i<=besti; i++) {
        if (theta[i] == THETA_DISTRACTOR)
            d++;
        else if (theta[i] == THETA_CONFLICT)
            c++;
        else
            m++;
    }
    if (p_nconflict) *p_nconflict = c;
    if (p_ndistractor) *p_ndistractor = d;
    if (p_nmatch) *p_nmatch = m;
}


void verify_hit(const startree_t* skdt, int index_cutnside, MatchObj* mo,
                const sip_t* sip, const verify_field_t* vf,
                double pix2, double distractors,
                double fieldW, double fieldH,
                double logbail, double logaccept, double logstoplooking,
                anbool do_gamma, anbool fake_match) {
    int i,j;
    double* fieldcenter;
    double fieldr2;
    double effA, K, worst;
    int besti;
    int* theta = NULL;
    double* allodds = NULL;
    sip_t thewcs;
    int ibad, igood;
    double* refxyz = NULL;
    int* sweep = NULL;
    verify_t the_v;
    verify_t* v = &the_v;
    int NRimage;
    int ibailed, istopped;

    assert(mo->wcs_valid || sip);
    assert(isfinite(logaccept));
    assert(isfinite(logbail));

    memset(v, 0, sizeof(verify_t));

    if (sip)
        v->wcs = sip;
    else {
        sip_wrap_tan(&mo->wcstan, &thewcs);
        v->wcs = &thewcs;
    }

    // center and radius of the field in xyz space:
    fieldcenter = mo->center;
    fieldr2 = square(mo->radius);
    debug("Field center %g,%g,%g, radius2 %g\n", fieldcenter[0], fieldcenter[1], fieldcenter[2], fieldr2);

    if (log_get_level() >= LOG_VERB) {
        double ra,dec, r;
        xyzarr2radecdeg(fieldcenter, &ra, &dec);
        r = distsq2deg(fieldr2);
        debug("Field center RA,Dec %g,%g, radius %g deg\n", ra, dec, r);
    }

    // find index stars and project them into pixel coordinates.
    /*
     verify_get_index_stars(fieldcenter, fieldr2, skdt, sip, &(mo->wcstan),
     fieldW, fieldH, NULL, &refxy, &starids, &NR);
     */
    /*
     Gotta be a bit careful with reference stars:
	 
     We want to be able to return a list of all the reference
     stars in the image, but during the verification process we
     want to apply some filtering of reference stars.  We
     therefore keep an int array ("refperm") of indices into the
     arrays of reference star quantities.  There are "NR" good
     stars, but "NRall" in total.  Thus operations on all the
     stars must go to "NRall" in the original arrays, but
     operations on good stars must go to "NR", using "refperm" to
     redirect.

     This means that "refperm" should remain a permutation array (ie,
     no duplicates), and each value should be less than "NRall"; when
     filtering out an index, it should get moved to the part of the
     array between "NR" and "NRall".  We use the "badguys" array to
     hold these indices temporarily.
     */
    assert(skdt->sweep);
    // Find all index stars within the bounding circle of the field.
    startree_search_for(skdt, fieldcenter, fieldr2, &refxyz, NULL, &v->refstarid, &v->NRall);
    debug2("%i reference stars in the bounding circle\n", v->NRall);
    if (!refxyz) {
        // no stars in range.
        logverb("No reference stars in the bounding circle\n");
        goto bailout;
    }
    //logverb("Found %i reference stars in the bounding circle\n", v->NRall);
    // Find index stars within the rectangular field.
    v->refxy = malloc(v->NRall * 2 * sizeof(double));
    v->refperm = malloc(v->NRall * sizeof(int));
    igood = 0;
    for (i=0; i<v->NRall; i++) {
        if (!sip_xyzarr2pixelxy(v->wcs, refxyz+i*3, v->refxy+i*2, v->refxy+i*2 +1) ||
            !sip_pixel_is_inside_image(v->wcs, v->refxy[i*2], v->refxy[i*2+1])) {
            continue;
        }
        v->refperm[igood] = i;
        igood++;
    }
    v->NR = igood;
    // We sort of want to forget about stars not within the image...
    // but we don't want to change NRall...
    NRimage = v->NR;
    // NOTE that at this point, v->refperm elements past NRimage are invalid
    // (ie, may contain repeats)

    // Sort by sweep #.
    // Each index star has a "sweep number" assigned during index building;
    // it roughly represents a local brightness ordering.  Use this to sort the
    // index stars.
    // (NOTE that here we do want "sweep" to be size "NRall"; only the
    // bottom "NRimage" of the "refperm" array will be accessed in the
    // permuted_sort below, so none of
    // the elements between NRimage and NRall will be touched.)
    sweep = malloc(v->NRall * sizeof(int));
    for (i=0; i<v->NRall; i++)
        sweep[i] = skdt->sweep[v->refstarid[i]];
    // Note here that we're passing in an existing permutation array; it
    // gets re-permuted during this call.
    permuted_sort(sweep, sizeof(int), compare_ints_asc, v->refperm, v->NR);
    free(sweep);
    sweep = NULL;
    debug2("Found %i reference stars.\n", v->NR);

    // "refstarids" are indices into the star kdtree and could be used to
    // retrieve "tag-along" data with, eg, startree_get_data_column().

    v->badguys = malloc(v->NR * sizeof(int));

    // remove reference stars that are part of the quad.
    if (!fake_match) {
        ibad = 0;
        igood = 0;
        for (i=0; i<v->NR; i++) {
            anbool inquad = FALSE;
            int ri = v->refperm[i];
            for (j=0; j<mo->dimquads; j++) {
                if (v->refstarid[ri] == mo->star[j]) {
                    inquad = TRUE;
                    //debug2("Skipping ref star index %i, starid %i: quad star %i\n", ri, v->refstarid[ri], j);
                    v->badguys[ibad] = ri;
                    ibad++;
                    break;
                }
            }
            if (inquad)
                continue;
            v->refperm[igood] = ri;
            igood++;
        }
        // remember the bad guys
        memcpy(v->refperm + igood, v->badguys, ibad * sizeof(int));
        v->NR = igood;
        debug2("After removing stars in the quad: %i reference stars.\n", v->NR);
    }
	
    if (!v->NR) {
        logverb("After removing quad stars: no reference stars\n");
        goto bailout;
    }

    ///// FIXME -- we could compute the RoR and search for ref stars
    // based on the quad center and RoR rather than the image center
    // and image radius.

    if (!fake_match) {
        verify_apply_ror(v, index_cutnside, mo,
                         vf, pix2, distractors, fieldW, fieldH,
                         do_gamma, fake_match,
                         &effA, NULL, NULL);
        if (!v->NR) {
            logerr("After applying ROR, NR = 0!\n");
            goto bailout;
        }
    } else {
        verify_get_test_stars(v, vf, mo, pix2, do_gamma, fake_match);
        effA = fieldW * fieldH;
        debug2("Number of test stars: %i\n", v->NT);
    }
    if (!v->NR || !v->NT) {
        logverb("After applying RoR, NR=%i, NT=%i\n", v->NR, v->NT);
        goto bailout;
    }

    worst = -LARGE_VAL;
    K = real_verify_star_lists(v, effA, distractors,
                               logbail, logstoplooking, &besti, &allodds, &theta, &worst,
                               &ibailed, &istopped);
    mo->logodds = K;
    mo->worstlogodds = worst;
    // NTall so that caller knows how big 'etheta' is.
    mo->nfield = v->NTall;
    // NRimage: only the stars inside the image bounds.
    mo->nindex = NRimage;

    if (log_get_level() >= LOG_ALL) {
        int nm, nc, nd;
        verify_count_hits(theta, besti, &nm, &nc, &nd);
        debug("verify: logodds %g, %i matches, %i conflicts, %i distractors after %i field objects.\n",
              K, nm, nc, nd, besti);
    }

    if (K >= logaccept) {
        int ri, ti;
        int* etheta;
        double* eodds;
        int nm, nc, nd;
        verify_count_hits(theta, besti, &nm, &nc, &nd);
        mo->nmatch = nm;
        mo->nconflict = nc;
        mo->ndistractor = nd;

        fixup_theta(theta, allodds, ibailed, istopped, v, besti, NRimage, refxyz,
                    &etheta, &eodds);

        // Reinsert the matched quad...
        if (!fake_match) {
            for (j=0; j<mo->dimquads; j++) {
                // the ref star should have been eliminated, so it
                // should be in the "bad" part of the array, but
                // search the whole thing anyway.
                for (i=0; i<NRimage; i++) {
                    ri = i;
                    if (v->refstarid[ri] == mo->star[j]) {
                        ti = mo->field[j];
                        assert(etheta[ti] == THETA_FILTERED);
                        etheta[ti] = ri;
                        eodds[ti] = LARGE_VAL;
                        debug2("Matched ref index %i (star %i) to test index %i; ref pos=(%.1f, %.1f), test pos=(%.1f, %.1f)\n",
                               ri, v->refstarid[ri], ti, v->refxy[ri*2+0], v->refxy[ri*2+1], v->testxy[ti*2+0], v->testxy[ti*2+1]);
                        break;
                    }
                }
            }
        }

        if (DEBUGVERIFY) {
            debug2("\n");
            for (i=0; i<v->NTall; i++) {
                debug2("ETheta[%i] = %i", i, etheta[i]);
                if (etheta[i] < 0) {
                    debug2(" (w=%g)\n", verify_logodds_to_weight(eodds[i]));
                    continue;
                }
                ri = etheta[i];
                ti = i;
                debug2(" (starid %i), testxy=(%.1f, %.1f), refxy=(%.1f, %.1f), logodds=%g, w=%g\n",
                       v->refstarid[ri], v->testxy[ti*2+0], v->testxy[ti*2+1], v->refxy[ri*2+0], v->refxy[ri*2+1],
                       eodds[i], verify_logodds_to_weight(eodds[i]));
            }
        }

        mo->theta = etheta;
        mo->matchodds = eodds;
        mo->refxyz = refxyz;
        refxyz = NULL;
        mo->refxy = v->refxy;
        v->refxy = NULL;
        mo->refstarid = v->refstarid;
        v->refstarid = NULL;
        mo->testperm = v->testperm;
        v->testperm = NULL;

        matchobj_compute_derived(mo);
    }

 cleanup:
    free(refxyz);
    free(theta);
    free(allodds);
    free(v->testperm);
    free(v->testsigma);
    free(v->tbadguys);
    free(v->refperm);
    free(v->refxy);
    free(v->refstarid);
    free(v->badguys);
    return;

 bailout:
    set_null_mo(mo);
    // uh oh, spaghetti-code-oh!
    goto cleanup;
}

// Free the things we added to this mo.
void verify_free_matchobj(MatchObj* mo) {
    free(mo->refxyz);
    free(mo->refstarid);
    free(mo->refxy);
    free(mo->theta);
    free(mo->matchodds);
    free(mo->testperm);
    mo->testperm = NULL;
    mo->refxyz = NULL;
    mo->refstarid = NULL;
    mo->refxy = NULL;
    mo->theta = NULL;
    mo->matchodds = NULL;
}

void verify_matchobj_deep_copy(const MatchObj* mo, MatchObj* dest) {
    if (mo->refxyz) {
        dest->refxyz = malloc(mo->nindex * 3 * sizeof(double));
        memcpy(dest->refxyz, mo->refxyz, mo->nindex * 3 * sizeof(double));
    }
    if (mo->refxy) {
        dest->refxy = malloc(mo->nindex * 2 * sizeof(double));
        memcpy(dest->refxy, mo->refxy, mo->nindex * 2 * sizeof(double));
    }
    if (mo->refstarid) {
        dest->refstarid = malloc(mo->nindex * sizeof(int));
        memcpy(dest->refstarid, mo->refstarid, mo->nindex * sizeof(int));
    }
    if (mo->matchodds) {
        dest->matchodds = malloc(mo->nfield * sizeof(double));
        memcpy(dest->matchodds, mo->matchodds, mo->nfield * sizeof(double));
    }
    if (mo->theta) {
        dest->theta = malloc(mo->nfield * sizeof(int));
        memcpy(dest->theta, mo->theta, mo->nfield * sizeof(int));
    }
}

double verify_logodds_to_weight(double lodds) {
    if (lodds > 40.)
        return 1.0;
    if (lodds < -700)
        return 0.0;
    return exp(lodds) / (1.0 + exp(lodds));
}


double verify_star_lists(double* refxys, int NR,
                         const double* testxys, const double* testsigma2s, int NT,
                         double effective_area,
                         double distractors,
                         double logodds_bail,
                         double logodds_stoplooking,
                         int* p_besti,
                         double** p_all_logodds, int** p_theta,
                         double* p_worstlogodds,
                         int** p_testperm) {
    double X;
    verify_t v;
    double* eodds;
    int* etheta;
    int ibailed, istopped;
    int besti;
    int* theta;
    double* allodds;

    memset(&v, 0, sizeof(verify_t));
    v.NRall = v.NR = NR;
    v.NTall = v.NT = NT;
    // discard const here...
    v.refxy = (double*)refxys;
    v.testxy = (double*)testxys;
    v.testsigma = (double*)testsigma2s;

    v.refperm = permutation_init(NULL, NR);
    v.testperm = permutation_init(NULL, NT);

    X = real_verify_star_lists(&v, effective_area, distractors,
                               logodds_bail, logodds_stoplooking, &besti,
                               &allodds, &theta,
                               p_worstlogodds, &ibailed, &istopped);
    fixup_theta(theta, allodds, ibailed, istopped, &v, besti, NR, NULL,
                &etheta, &eodds);
    free(theta);
    free(allodds);

    if (p_all_logodds)
        *p_all_logodds = eodds;
    else
        free(eodds);
    if (p_theta)
        *p_theta = etheta;
    else
        free(etheta);

    if (p_besti)
        *p_besti = besti;

    if (p_testperm)
        *p_testperm = v.testperm;
    else
        free(v.testperm);

    free(v.refperm);
    free(v.badguys);
    return X;
}









double verify_star_lists_ror(double* refxys, int NR,
                             const double* testxys, const double* testsigma2s, int NT,
                             double pix2, double gamma,
                             const double* qc, double Q2,
                             double W, double H,
                             double distractors,
                             double logodds_bail,
                             double logodds_stoplooking,
                             int* p_besti,
                             double** p_all_logodds, int** p_theta,
                             double* p_worstlogodds,
                             int** p_testperm, int** p_refperm) {
    double X;
    verify_t v;
    double* eodds = NULL;
    int* etheta = NULL;
    int ibailed, istopped;
    int besti = -1;
    int* theta = NULL;
    double* allodds = NULL;
    // RoR
    double ror2;
    int igood, ibad;
    int NB = 100;
    int NBx, NBy;
    double bx0, by0;
    double stepx, stepy;
    int i, j;
    int Ngood;
    double effective_area;

    memset(&v, 0, sizeof(verify_t));
    v.NRall = v.NR = NR;
    v.NTall = v.NT = NT;
    v.refxy = refxys;
    // instead of verify_get_test_stars()...
    // (so we don't do:
    // --dedup
    // --remove quad stars
    // --uniformize
    // )

    // discard const here...
    v.testxy = (double*)testxys;
    v.testsigma = (double*)testsigma2s;
    v.refperm = permutation_init(NULL, NR);
    v.testperm = permutation_init(NULL, NT);
    v.tbadguys = malloc(v.NTall * sizeof(int));
    v.badguys = malloc(v.NRall * sizeof(int));

    ror2 = verify_get_ror2(Q2, W*H, distractors, NR, pix2);
    logverb("RoR: %g\n", sqrt(ror2));

    // Remove test stars outside the RoR.
    igood = ibad = 0;
    for (i=0; i<v.NT; i++) {
        int ti = v.testperm[i];
        double r2 = distsq(qc, v.testxy + 2*ti, 2);
        if (r2 < ror2) {
            v.testperm[igood] = ti;
            igood++;
        } else {
            v.tbadguys[ibad] = ti;
            ibad++;
        }
    }
    v.NT = igood;
    // remember the bad guys
    memcpy(v.testperm + igood, v.tbadguys, ibad * sizeof(int));
    logverb("Test stars in RoR: %i of %i\n", v.NT, v.NTall);

    // Count good bins to find effective area...
    NBx = ceil((double)W / sqrt(W*H) * sqrt(NB));
    NBy = ceil((double)H / sqrt(W*H) * sqrt(NB));
    NB = NBx * NBy;
    stepx = (double)W / (double)NBx;
    stepy = (double)H / (double)NBy;
    bx0 = stepx/2.0;
    by0 = stepy/2.0;
    Ngood = 0;
    for (i=0; i<NBy; i++) {
        double bxy[2];
        bxy[1] = by0 + i*stepy;
        for (j=0; j<NBx; j++) {
            double r2;
            bxy[0] = bx0 + j*stepx;
            r2 = distsq(bxy, qc, 2);
            if (r2 < ror2)
                Ngood++;
        }
    }
    effective_area = W*H * (double)Ngood / (double)NB;
    logverb("Good bins: %i / %i; effA %g of %g\n", Ngood, NB, W*H, effective_area);

    // Remove ref stars outside RoR.
    igood = ibad = 0;
    for (i=0; i<v.NR; i++) {
        int ri = v.refperm[i];
        if (distsq(qc, v.refxy + 2*ri, 2) < ror2) {
            v.refperm[igood] = ri;
            igood++;
        } else {
            v.badguys[ibad] = ri;
            ibad++;
        }
    }
    // remember the bad guys
    memcpy(v.refperm + igood, v.badguys, ibad * sizeof(int));
    v.NR = igood;
    logverb("Ref stars in RoR: %i of %i\n", v.NR, v.NRall);

    if (v.NR) {
        X = real_verify_star_lists(&v, effective_area, distractors,
                                   logodds_bail, logodds_stoplooking, &besti,
                                   &allodds, &theta,
                                   p_worstlogodds, &ibailed, &istopped);
        fixup_theta(theta, allodds, ibailed, istopped, &v, besti, NR, NULL,
                    &etheta, &eodds);
        free(theta);
        free(allodds);

        if (p_all_logodds)
            *p_all_logodds = eodds;
        else
            free(eodds);
        if (p_theta)
            *p_theta = etheta;
        else
            free(etheta);

        if (p_besti)
            *p_besti = besti;

    } else {
        X = -LARGE_VAL;
    }


    if (p_testperm)
        *p_testperm = v.testperm;
    else
        free(v.testperm);


    if (p_refperm)
        *p_refperm = v.refperm;
    else
        free(v.refperm);

    free(v.badguys);
    free(v.tbadguys);
	
    return X;
}

