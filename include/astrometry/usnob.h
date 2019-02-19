/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef USNOB_H
#define USNOB_H

#include "astrometry/an-bool.h"
#include <stdint.h>

#include "astrometry/starutil.h"

#define USNOB_RECORD_SIZE 80

#define USNOB_SURVEY_POSS_I_O 0
#define USNOB_SURVEY_POSS_I_E 1
#define USNOB_SURVEY_POSS_II_J 2
#define USNOB_SURVEY_POSS_II_F 3
#define USNOB_SURVEY_SERC_J 4
#define USNOB_SURVEY_SERC_EJ 4
#define USNOB_SURVEY_ESO_R 5
#define USNOB_SURVEY_SERC_ER 5
#define USNOB_SURVEY_AAO_R 6
#define USNOB_SURVEY_POSS_II_N 7
#define USNOB_SURVEY_SERC_I 8
#define USNOB_SURVEY_SERC_I_OR_POSS_II_N 9

struct observation {
    // 0 to 99.99 (m:4)
    float mag;

    // field number in the original survey. 1-937 (F:3)
    int16_t field;

    // the original survey. (S:1)
    // (eg USNOB_SURVEY_POSS_I_O)
    uint8_t survey;

    // star/galaxy estimate.  0=galaxy, 11=star. 19=no value computed.
    //     (GG:2)
    // (but note, in fact values 12, 13, 14, 15 and possibly others exist
    //  in the data files as well!)
    uint8_t star_galaxy;

    // [degrees] (R:4)
    float xi_resid;

    // [degrees] (r:4)
    float eta_resid;

    // source of photometric calibration: (C:1)
    //  0=bright photometric standard on this plate
    //  1=faint pm standard on this plate
    //  2=faint " " one plate away
    //  etc
    uint8_t calibration;

    // back-pointer to PMM file. (i:7)
    int32_t pmmscan;
};

#define OBS_BLUE1 0
#define OBS_RED1  1
#define OBS_BLUE2 2
#define OBS_RED2  3
#define OBS_N     4

struct usnob_entry {
    // (numbers in brackets are number of digits used in USNOB format)
    // [degrees] (a:9)
    double ra;
    // [degrees] (s:8)
    double dec;

    // [degrees] (u:3)
    float sigma_ra;
    // [degrees] (v:3)
    float sigma_dec;

    // [degrees] (Q:1)
    float sigma_ra_fit;
    // [degrees] (R:1)
    float sigma_dec_fit;

    // proper motion
    // [arcsec/yr] (A:3) 
    float pm_ra;
    // [arcsec/yr] (S:3)
    float pm_dec;

    // [arcsec/yr] (x:3)
    float sigma_pm_ra;
    // [arcsec/yr] (y:3)
    float sigma_pm_dec;

    // motion probability. (P:1)
    float pm_prob;

    // [yr] 1950-2050. (e:3)
    float epoch;

    // number of detections; (M:1)
    // M=0 means Tycho-2 star.  In this case, NONE of the other fields
    //                          in the struct can be trusted!  The USNOB
    //                          compilers used a different (and undocumented)
    //                          format to store Tycho-2 stars.
    // M=1 means it's a reject USNOB star.
    // M>=2 means it's a valid USNOB star.
    uint8_t ndetections;

    anbool diffraction_spike;
    anbool motion_catalog;
    // YS4.0 correlation
    anbool ys4;

    // astrometry.net diffraction detection: 0, 1=spike, 2=halo
    uint8_t an_diffraction_spike;

    // this is our identifier; it's not in the USNO-B files.
    // it allows us to point back to the USNO-B source.
    // top byte: [0,180): south-polar distance slice.
    // bottom 24 bits: [0, 12,271,141): index within slice.
    uint32_t usnob_id;

    // this is a "staging" variable used by the FITS i/o routines.
    uint8_t flags;

    // the observations for this object.  These are stored in a fixed
    // order (same as the raw USNOB data):
    //   obs[OBS_BLUE1] is the "first-epoch (old) blue" observation,
    //   obs[OBS_RED2]  is the "second-epoch (new) red" observation
    //
    // Note that many objects have fewer than five observations.  To check
    // whether an observation exists, check the "field" value: all valid
    // observations have non-zero values.
    struct observation obs[5];
};
typedef struct usnob_entry usnob_entry;

int usnob_get_slice(usnob_entry* entry);

int usnob_get_index(usnob_entry* entry);

int usnob_parse_entry(unsigned char* line, usnob_entry* usnob);

// Returns 0 if this entry has a measured blue magnitude and sets 'mag';
// returns -1 if it has no blue measurements.
int usnob_get_blue_mag(usnob_entry* entry, float* mag);
int usnob_get_red_mag(usnob_entry* entry, float* mag);
int usnob_get_infrared_mag(usnob_entry* entry, float* mag);

unsigned char usnob_get_survey_band(int survey);

// returns 1 if the observation is first-epoch
//         2 if the observation is second-epoch
//        -1 on error.
int unsob_get_survey_epoch(int survey, int obsnum);

/*
 Returns TRUE if this entry is a true USNOB star, not a Tycho-2 or reject.
 (This doesn't check diffraction flags, just the "M" / "ndetection" field).
 */
anbool usnob_is_usnob_star(usnob_entry* entry);

/*
 Returns TRUE if the given observation contains real data.
 (Note that "usnob_is_usnob_star" must pass for this to be valid)
 */
anbool usnob_is_observation_valid(struct observation* obs);

/*
 Returns TRUE if the given bandpass (emulsion) is "blue" (band is 'O' or 'J').
 */
anbool usnob_is_band_blue(unsigned char band);

/* Returns TRUE if the given observation comes from a blue emulsion. */
anbool usnob_is_observation_blue(struct observation* obs);

/*
 Returns TRUE if the given bandpass (emulsion) is "red" (band is 'E' or 'F').
 */
anbool usnob_is_band_red(unsigned char band);

/* Returns TRUE if the given observation comes from a red emulsion. */
anbool usnob_is_observation_red(struct observation* obs);

/*
 Returns TRUE if the given bandpass (emulsion) is "infrared" (band is 'N')
 */
anbool usnob_is_band_ir(unsigned char band);

/* Returns TRUE if the given observation comes from an infrared emulsion. */
anbool usnob_is_observation_ir(struct observation* obs);

#endif
