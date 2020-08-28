/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef MATCHOBJ_H
#define MATCHOBJ_H

#include <stdint.h>

#include "astrometry/starutil.h"
#include "astrometry/sip.h"
#include "astrometry/bl.h"
#include "astrometry/index.h"
#include "astrometry/an-bool.h"

struct match_struct {
    unsigned int quadno;
    unsigned int star[DQMAX];
    unsigned int field[DQMAX];
    uint64_t ids[DQMAX];
    // actually code error ^2.
    float code_err;

    // Pixel positions of the quad stars.
    double quadpix[2 * DQMAX];

    // Un-distorted pixel positions of quad stars.
    double quadpix_orig[2 * DQMAX];

    // Star positions of the quad stars.
    double quadxyz[3 * DQMAX];

    uint8_t dimquads;

    // the center of the field in xyz coords
    double center[3];
    // radius of the bounding circle, in distance on the unit sphere.
    double radius;
    // radius of the bounding circle in degrees
    double radius_deg;

    // WCS params
    anbool wcs_valid;
    tan_t wcstan;

    // arcseconds per pixel; computed: scale=3600*sqrt(abs(det(wcstan.cd)))
    double scale;
    
    // How many quads were matched to this single quad (not a running total)
    // (only counts a single ABCD permutation)
    int16_t quad_npeers;

    int nmatch;
    int ndistractor;
    int nconflict;
    int nfield;
    int nindex;

    // nbest = besti+1 = nmatch + ndistractor + nconflict <= nfield
    int nbest;

    float logodds;

    float worstlogodds;

    // how many other matches agreed with this one *at the time it was found*
    int16_t nagree;

    int fieldnum;
    int fieldfile;
    int16_t indexid;
    int16_t healpix;
    int16_t hpnside;

    char fieldname[32];

    anbool parity;

    // how many field quads did we try before finding this one?
    int quads_tried;
    // how many matching quads from the index did we find before this one?
    int quads_matched;
    // how many matching quads had the right scale?
    int quads_scaleok;
    // how many field objects did we have to look at?
    //  (this isn't stored in the matchfile, it's max(field))
    int objs_tried;
    // how many matches have we run verification on?
    int nverified;
    // how many seconds of CPU time have we spent on this field?
    float timeused;

    // stuff used by onefield...
    // tweaked-up WCS.
    sip_t* sip;
    // RA,Dec of reference stars; length is "nindex".
    double* refradec;
    // for correspondence file we need a copy of the field! (star x,y positions)
    double* fieldxy;
    // + original (un-distorted) x,y coords
    double* fieldxy_orig;

    bl* tagalong;
    bl* field_tagalong;

    // in arcsec.
    double index_jitter;

    index_t* index;

    // from verify.c: correspondences between index and image stars.
    // length of this array is "nfield".
    // Element i corresponds to field star 'i'; theta[i] is the reference star 
    // that matched, as an index in the "refxyz", "refxy", and "refstarid" arrays;
    // OR one of the (negative) special values THETA_DISTRACTOR, THETA_CONFLICT, THETA_FILTERED.
    int* theta;

    // log-odds that the matches in 'theta' are correct;
    // this array is parallel to 'theta' so has length "nfield".
    // Star that were unmatched (THETA_DISTRACTOR, etc) have -inf;
    // Stars that are part of the matched quad have +inf.
    // Normal matches contain log-odds of the match: log(p(fg)/p(bg));
    // see verify.h : verify_logodds_to_weight(x) to convert to a weight
    // that is 0 for definitely not-a-match and 1 for definitely a match.
    double* matchodds;

    // the order in which test stars were tried during verification
    int* testperm;

    // the stuff we discover about the reference stars during verify().
    // These arrays have length "nindex"; they include all reference
    // stars that should appear in the image.
    // refxyz[i*3] -- x,y,z unit sphere position of this star; xyz2radec
    double* refxyz;
    // refxy[i*2] -- pixel x,y position (according to the *un-tweaked* WCS)
    double* refxy;
    // refstarid[i] -- index in the star kdtree (ie, can be used with startree_get_data_column as 'indices')
    int* refstarid;

};
typedef struct match_struct MatchObj;

void matchobj_compute_overlap(MatchObj* mo);

// compute all derived fields.
void matchobj_compute_derived(MatchObj* mo);

// Returns the name of the index that produced this match.
const char* matchobj_get_index_name(MatchObj* mo);

void matchobj_log_hit_miss(int* theta, int* testperm, int nbest, int nfield, int loglevel, const char* prefix);

char* matchobj_hit_miss_string(int* theta, int* testperm, int nbest,
                               int nfield, char* target);

void matchobj_print(MatchObj* mo, int loglvl);

#endif
