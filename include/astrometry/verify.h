/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef VERIFY_H
#define VERIFY_H

#include "astrometry/kdtree.h"
#include "astrometry/matchobj.h"
#include "astrometry/bl.h"
#include "astrometry/starkd.h"
#include "astrometry/sip.h"
#include "astrometry/bl.h"
#include "astrometry/starxy.h"

struct verify_field_t {
    const starxy_t* field;
    // this copy is normal.
    double* xy;
    // this copy is permuted by the kdtree
    double* fieldcopy;
    kdtree_t* ftree;

    // should this field be spatially uniformized at the index's scale?
    anbool do_uniformize;
    // should this field be de-duplicated (have nearby sources removed)?
    anbool do_dedup;
    // apply radius-of-relevance filtering
    anbool do_ror;
};
typedef struct verify_field_t verify_field_t;


/*
 This function must be called once for each field before verification
 begins.  We build a kdtree out of the field stars (in pixel space)
 which will be used during deduplication.
 */
verify_field_t* verify_field_preprocess(const starxy_t* fieldxy);

/*
 This function must be called after all verification calls for a field
 are finished; we clean up the data structures we created in the
 verify_field_preprocess() function.
 */
void verify_field_free(verify_field_t* vf);




void verify_count_hits(int* theta, int besti, int* p_nmatch, int* p_nconflict, int* p_ndistractor);

void verify_wcs(const startree_t* skdt,
                int index_cutnside,
                const sip_t* sip,
                const verify_field_t* vf,
                double verify_pix2,
                double distractors,
                double fieldW,
                double fieldH,
                double logratio_tobail,
                double logratio_toaccept,
                double logratio_tostoplooking,

                double* logodds,
                int* nfield, int* nindex,
                int* nmatch, int* nconflict, int* ndistractor
                // int** theta ?
                );

/*
 Uses the following entries in the "mo" struct:
 -wcs_valid
 -wcstan
 -center
 -radius
 -field[]
 -star[]
 -dimquads

 Sets the following:
 -nfield
 -noverlap
 -nconflict
 -nindex
 -(matchobj_compute_derived() values)
 -logodds
 -corr_field
 -corr_index
 */
void verify_hit(const startree_t* skdt,
                int index_cutnside,
                // input/output param.
                MatchObj* mo,
                const sip_t* sip, // if non-NULL, verify this SIP WCS.
                const verify_field_t* vf,
                double verify_pix2,
                double distractors,
                double fieldW,
                double fieldH,
                double logratio_tobail,
                double logratio_toaccept,
                double logratio_tostoplooking,
                anbool distance_from_quad_bonus,
                anbool fake_match);

// Distractor
#define THETA_DISTRACTOR -1
// Conflict
#define THETA_CONFLICT -2
// Filtered out
#define THETA_FILTERED -3
// Not examined because the bail-out threshold was reached.
#define THETA_BAILEDOUT -4
// Not examined because the stop-looking threshold was reached.
#define THETA_STOPPEDLOOKING -5

/*
 void verify_apply_ror(double* refxy, int* starids, int* p_NR,
 int index_cutnside,
 MatchObj* mo,
 const verify_field_t* vf,
 double pix2,
 double distractors,
 double fieldW,
 double fieldH,
 anbool do_gamma, anbool fake_match,
 double** p_testxy, double** p_sigma2s,
 int* p_NT, int** p_perm, double* p_effA,
 int* p_uninw, int* p_uninh);
 */

/**
 Returns the best log-odds encountered.
 */
double verify_star_lists(double* refxys, int NR,
                         const double* testxys, const double* testsigma2s, int NT,
                         double effective_area,
                         double distractors,
                         double logodds_bail,
                         double logodds_accept,
                         int* p_besti,
                         double** p_all_logodds, int** p_theta,
                         double* p_worstlogodds,
                         int** p_testperm);

void verify_get_uniformize_scale(int cutnside, double scale, int W, int H, int* uni_nw, int* uni_nh);

void verify_uniformize_field(const double* xy, int* perm, int N,
                             double fieldW, double fieldH,
                             int nw, int nh,
                             int** p_bincounts,
                             int** p_binids);

double* verify_uniformize_bin_centers(double fieldW, double fieldH,
                                      int nw, int nh);

void verify_get_quad_center(const verify_field_t* vf, const MatchObj* mo, double* centerpix,
                            double* quadr2);

/*
 int verify_get_test_stars(const verify_field_t* vf, MatchObj* mo,
 double pix2, anbool do_gamma,
 anbool fake_match,
 double** p_sigma2s, int** p_perm);
 */

void verify_get_index_stars(const double* fieldcenter, double fieldr2,
                            const startree_t* skdt, const sip_t* sip, const tan_t* tan,
                            double fieldW, double fieldH,
                            double** p_indexradec,
                            double** p_indexpix, int** p_starids, int* p_nindex);

/*
 anbool* verify_deduplicate_field_stars(const verify_field_t* vf, double* sigma2s, double nsigmas);
 */
/*
 double* verify_compute_sigma2s_arr(const double* xy, int NF,
 const double* qc, double Q2,
 double verify_pix2, anbool do_gamma);
 */

// For use with matchobj.h : matchodds
double verify_logodds_to_weight(double lodds);

void verify_free_matchobj(MatchObj* mo);

void verify_matchobj_deep_copy(const MatchObj* mo, MatchObj* dest);

double verify_get_ror2(double Q2, double area,
                       double distractors, int NR, double pix2);



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
                             int** p_testperm, int** p_refperm);

#endif
