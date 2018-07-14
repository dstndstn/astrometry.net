/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "usnob.h"
#include "an-endian.h"

int unsob_get_survey_epoch(int survey, int obsnum) {
    switch (survey) {
    case USNOB_SURVEY_POSS_I_O:
    case USNOB_SURVEY_POSS_I_E:
        return 1;

    case USNOB_SURVEY_POSS_II_J:
    case USNOB_SURVEY_POSS_II_F:
    case USNOB_SURVEY_POSS_II_N:
    case USNOB_SURVEY_SERC_J:
        //case USNOB_SURVEY_SERC_EJ:
    case USNOB_SURVEY_AAO_R:
    case USNOB_SURVEY_SERC_I:
    case USNOB_SURVEY_SERC_I_OR_POSS_II_N:
        return 2;

    case USNOB_SURVEY_ESO_R:
        //case USNOB_SURVEY_SERC_ER:
        {
            if (obsnum == 1)
                return 1;
            if (obsnum == 3)
                return 2;
            return -1;
        }
    default:
        return -1;
    }
}

unsigned char usnob_get_survey_band(int survey) {
    // from Tables 1 and 3 (esp footnote h) of the USNO-B paper.
    switch (survey) {
    case USNOB_SURVEY_POSS_I_O:
        return 'O'; // blue (350-500 mm)
    case USNOB_SURVEY_POSS_I_E:
        return 'E'; // red  (620-670 mm)
    case USNOB_SURVEY_POSS_II_J:
    case USNOB_SURVEY_SERC_J:
        //case USNOB_SURVEY_SERC_EJ:
        return 'J'; // blue (385-540 mm)
    case USNOB_SURVEY_POSS_II_F:
    case USNOB_SURVEY_ESO_R:
        //case USNOB_SURVEY_SERC_ER:
    case USNOB_SURVEY_AAO_R:
        return 'F'; // red  (590, 610 or 630-690 mm)
    case USNOB_SURVEY_POSS_II_N:
    case USNOB_SURVEY_SERC_I:
    case USNOB_SURVEY_SERC_I_OR_POSS_II_N:
        return 'N'; // infrared (715 or 730-900 mm)
    default:
        return '\0';
    }
}

int usnob_get_blue_mag(usnob_entry* entry, float* mag) {
    float sum = 0.0;
    int n = 0;
    // Dumbass mag averaging.
    if (usnob_is_observation_valid(entry->obs + OBS_BLUE1)) {
        sum += entry->obs[OBS_BLUE1].mag;
        n++;
    }
    if (usnob_is_observation_valid(entry->obs + OBS_BLUE2)) {
        sum += entry->obs[OBS_BLUE2].mag;
        n++;
    }
    if (n == 0)
        return -1;
    *mag = sum / (double)n;
    return 0;
}
int usnob_get_red_mag(usnob_entry* entry, float* mag) {
    float sum = 0.0;
    int n = 0;
    // Dumbass mag averaging.
    if (usnob_is_observation_valid(entry->obs + OBS_RED1)) {
        sum += entry->obs[OBS_RED1].mag;
        n++;
    }
    if (usnob_is_observation_valid(entry->obs + OBS_RED2)) {
        sum += entry->obs[OBS_RED2].mag;
        n++;
    }
    if (n == 0)
        return -1;
    *mag = sum / (double)n;
    return 0;
}
int usnob_get_infrared_mag(usnob_entry* entry, float* mag) {
    if (usnob_is_observation_valid(entry->obs + OBS_N)) {
        *mag = entry->obs[OBS_N].mag;
        return 0;
    }
    return -1;
}


anbool usnob_is_usnob_star(usnob_entry* entry) {
    return (entry->ndetections >= 2);
}

anbool usnob_is_observation_valid(struct observation* obs) {
    return (obs->field > 0);
}

anbool usnob_is_band_blue(unsigned char band) {
    return (band == 'O' || band == 'J');
}

anbool usnob_is_band_red(unsigned char band) {
    return (band == 'E' || band == 'F');
}

anbool usnob_is_band_ir(unsigned char band) {
    return (band == 'N');
}

anbool usnob_is_observation_blue(struct observation* obs) {
    return usnob_is_band_blue(usnob_get_survey_band(obs->survey));
}

anbool usnob_is_observation_red(struct observation* obs) {
    return usnob_is_band_red(usnob_get_survey_band(obs->survey));
}

anbool usnob_is_observation_ir(struct observation* obs) {
    return usnob_is_band_ir(usnob_get_survey_band(obs->survey));
}

int usnob_get_slice(usnob_entry* entry) {
    return (entry->usnob_id >> 24) & 0xFF;
}

int usnob_get_index(usnob_entry* entry) {
    return (entry->usnob_id & 0x00ffffff);
}

int usnob_parse_entry(unsigned char* line, usnob_entry* usnob) {
    int obs;
    int A, S, P, i, j, M, R, Q, y, x, k, e, v, u;
    uint32_t ival;
    uint32_t* uline;

    uline = (uint32_t*)line;

    // bytes 0-3: uint, RA in units of 0.01 arcsec.
    ival = u32_letoh(uline[0]);
    if (ival > (100*60*60*360)) {
        fprintf(stderr, "USNOB: RA should be in [0, %u), but got %u.\n",
                100*60*60*360, ival);
        assert(ival <= (100*60*60*360));
        return -1;
    }
    usnob->ra = arcsec2deg(ival * 0.01);

    // bytes 4-7: uint, SPD (south polar distance) in units of 0.01 arcsec.
    ival = u32_letoh(uline[1]);
    assert(ival <= (100*60*60*180));
    // DEC = south polar distance - 90 degrees
    usnob->dec = arcsec2deg(ival * 0.01) - 90.0;

    // bytes 8-11: uint, packed in base-10:
    //    iPSSSSAAAA
    ival = u32_letoh(uline[2]);
    A = (ival % 10000);
    ival     /= 10000;
    S = (ival % 10000);
    ival     /= 10000;
    P = (ival % 10);
    ival     /= 10;
    i = (ival % 10);

    // A: mu_RA, in units of 0.002 arcsec per year, offset by
    //    -10 arcsec per year.
    // (rewrite in this form to avoid cancellation error for zero)
    usnob->pm_ra = 0.002 * (A - 5000); // -10.0 + (0.002 * A);

    // S: mu_SPD, in units of 0.002 arcsec per year, offset by
    //    -10 arcsec per year.
    // This is a derivative of SPD which is equal to a derivative of DEC.
    usnob->pm_dec = 0.002 * (S - 5000); //-10.0 + (0.002 * S);

    // P: total mu probability, in units of 0.1.
    usnob->pm_prob = 0.1 * P;

    // i: motion catalog flag: 0=no, 1=yes.
    assert((i == 0) || (i == 1));
    usnob->motion_catalog = i;

    // bytes 12-15: uint, packed in base-10:
    //     jMRQyyyxxx
    ival = u32_letoh(uline[3]);
    x = (ival % 1000);
    ival     /= 1000;
    y = (ival % 1000);
    ival     /= 1000;
    Q = (ival % 10);
    ival     /= 10;
    R = (ival % 10);
    ival     /= 10;
    M = (ival % 10);
    ival     /= 10;
    j = (ival % 10);

    // x: sigma_mu_RA, in units of 0.001 arcsec per year.
    usnob->sigma_pm_ra = 0.001 * x;

    // y: sigma_mu_SPD, in units of 0.001 arcsec per year.
    // Again, a derivative of SPD = derivate of DEC.
    usnob->sigma_pm_dec = 0.001 * y;

    // Q: sigma_RA_fit, in units of 0.1 arcsec.
    usnob->sigma_ra_fit = arcsec2deg(0.1 * Q);

    // R: sigma_SPD_fit, in units of 0.1 arcsec.
    usnob->sigma_dec_fit = arcsec2deg(0.1 * R);

    // M: number of detections; in [2, 5] for USNOB stars,
    //   1 for rejected USNOB stars,
    //   0 for Tycho-2 stars.
    usnob->ndetections = M;

    // j: diffraction spike flag: 0=no, 1=yes.
    assert((j == 0) || (j == 1));
    usnob->diffraction_spike = j;

    // bytes 16-19: uint, packed in base-10:
    //     keeevvvuuu
    ival = u32_letoh(uline[4]);
    u = (ival % 1000);
    ival     /= 1000;
    v = (ival % 1000);
    ival     /= 1000;
    e = (ival % 1000);
    ival     /= 1000;
    k = (ival % 10);

    // u: sigma_RA, in units of 0.001 arcsec.
    usnob->sigma_ra = arcsec2deg(0.001 * u);

    // v: sigma_SPD, in units of 0.001 arcsec.
    // Again, equal to sigma_DEC.
    usnob->sigma_dec = arcsec2deg(0.001 * v);

    // e: mean epoch, in 0.1 yr, offset by -1950.
    usnob->epoch = 1950.0 + 0.1 * e;

    // k: YS4.0 correlation flag: 0=no, 1=yes.
    // (if M==0, it's a Tycho star and this has a different meaning)
    assert((M == 0) || (k == 0) || (k == 1));
    usnob->ys4 = (k == 1) ? 1 : 0;

    for (obs=0; obs<5; obs++) {
        int G, S, F, m, C, r, R;

        // bytes 20-23, 24-27, ...: uint, packed in base-10:
        //     GGSFFFmmmm
        ival = u32_letoh(uline[5 + obs]);
        m = (ival % 10000);
        ival     /= 10000;
        F = (ival % 1000);
        ival     /= 1000;
        S = (ival % 10);
        ival     /= 10;
        G = (ival % 100);

        // m: magnitude, in units of 0.01 mag.
        usnob->obs[obs].mag = 0.01 * m;

        // F: field number in the original survey; 1-937.
        if (M >= 2)
            assert(F <= 937);
        usnob->obs[obs].field = F;

        // S: survey number of original survey.
        usnob->obs[obs].survey = S;

        // G: star-galaxy estimate.  0=galaxy, 11=star.
        if (M >= 2) {
            // Hmm, this is triggered by USNOB10/033/b0337.cat
            // byte offset 10282000, observation 2, and MANY others.
            /*
             assert(G <= 11 || G == 19);
             if ((G > 11) && (G != 19)) {
             fprintf(stderr, "USNOB: star/galaxy estimate should be in {[0, 11], 19}, but found %u.\n", G);
             }
             */
        }
        usnob->obs[obs].star_galaxy = G;

        // bytes 40-43, 44-47, ...: uint, packed in base-10:
        //     CrrrrRRRR
        ival = u32_letoh(uline[10 + obs]);
        R = (ival % 10000);
        ival     /= 10000;
        r = (ival % 10000);
        ival     /= 10000;
        C = (ival % 10);

        if (M >= 2 && F == 0) {
            // empty observation.
            usnob->obs[obs].xi_resid  = 0.0;
            usnob->obs[obs].eta_resid = 0.0;
        } else {
            // R: xi residual, in units of 0.01 arcsec, offset by -50 arcsec.
            usnob->obs[obs].xi_resid = arcsec2deg(0.01 * (R - 5000)); // (-50.0 + 0.01 * R);
            // r: eta residual, in units of 0.01 arcesc, offset by -50.
            usnob->obs[obs].eta_resid = arcsec2deg(0.01 * (r - 5000)); // (-50.0 + 0.01 * r);
        }

        // C: source of photometric calibration.
        usnob->obs[obs].calibration = C;

        // bytes 60-63, 64-67, ...: uint
        // index into PMM scan file.
        ival = u32_letoh(uline[15 + obs]);
        assert(ival <= 9999999);
        usnob->obs[obs].pmmscan = ival;
    }

    return 0;
}

