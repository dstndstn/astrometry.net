/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "astrometry/tycho2.h"
#include "astrometry/starutil.h"

static void grab_substring(char* dst, const char* src, int n) {
    memset(dst, 0, n+1);
    strncpy(dst, src, n);
}

static int parse_uint(const char* src, int n, unsigned int* data) {
    char buf[256];
    grab_substring(buf, src, n);
    if (sscanf(buf, " %u", data) != 1) {
        fprintf(stderr, "Tycho2: couldn't parse uint record: '%.*s'\n", n, buf);
        return -1;
    }
    return 0;
}

static int parse_double(const char* src, int n, double* data) {
    char buf[256];
    grab_substring(buf, src, n);
    if (sscanf(buf, " %lf", data) != 1) {
        fprintf(stderr, "Tycho2: couldn't parse double record: '%.*s'\n", n, buf);
        return -1;
    }
    return 0;
}

static int parse_optional_double(const char* src, int n, double* data, double defaultval) {
    char buf[256];
    int i;
    grab_substring(buf, src, n);
    for (i=0; i<n; i++)
        if (buf[i] != ' ')
            break;
    if (i == n) {
        *data = defaultval;
        return 0;
    }
    if (sscanf(buf, " %lf", data) != 1) {
        fprintf(stderr, "Tycho2: couldn't parse optional double record: '%.*s'\n", n, buf);
        return -1;
    }
    return 0;
}

static int parse_optional_uint(const char* src, int n, unsigned int* data, unsigned int defaultval) {
    char buf[256];
    int i;
    grab_substring(buf, src, n);
    for (i=0; i<n; i++)
        if (buf[i] != ' ')
            break;
    if (i == n) {
        *data = defaultval;
        return 0;
    }
    if (sscanf(buf, " %u", data) != 1) {
        fprintf(stderr, "Tycho2: couldn't parse optional uint record: '%.*s'\n", n, buf);
        return -1;
    }
    return 0;
}

int tycho2_guess_is_supplement(const char* line) {
    return (line[12] == '|' &&
            line[14] == '|' &&
            line[27] == '|' &&
            line[40] == '|' &&
            line[48] == '|' &&
            line[56] == '|' &&
            line[62] == '|' &&
            line[68] == '|' &&
            line[74] == '|' &&
            line[80] == '|' &&
            line[82] == '|' &&
            line[89] == '|' &&
            line[95] == '|' &&
            line[102] == '|' &&
            line[108] == '|' &&
            line[112] == '|' &&
            line[114]);
}

int tycho2_parse_entry(const char* line, tycho2_entry* entry) {
    int len;
    unsigned int t1, t2, t3;
    char pxflag, tycho1, posflag;
    double d1, d2, d3, d4, d5, d6, d7, d8, d9, d10;
    double d11, d12, d13, d14, d15, d16;
    unsigned int u1, u2, u3;
    int i;

    memset(entry, 0, sizeof(tycho2_entry));

    for (len=0; len<TYCHO_RECORD_SIZE_RAW; len++)
        if (!line[len]) break;
    if (len != TYCHO_RECORD_SIZE_RAW) {
        fprintf(stderr, "Tycho2: couldn't parse record: length %i, not %i.\n",
                len, TYCHO_RECORD_SIZE_RAW);
        return -1;
    }

    if (parse_uint(line + 0, 4, &t1) ||
        parse_uint(line + 5, 5, &t2) ||
        parse_uint(line + 11, 1, &t3)) {
        return -1;
    }
    assert(t1 >= 1);
    assert(t1 <= 9537);
    assert(t2 >= 1);
    assert(t2 <= 12121);
    assert(t3 >= 1);
    assert(t3 <= 3);

    entry->tyc1 = t1;
    entry->tyc2 = t2;
    entry->tyc3 = t3;

    pxflag = line[13];
    switch (pxflag) {
    case ' ':
        break;
    case 'P':
        entry->photo_center = TRUE;
        break;
    case 'X':
        entry->no_motion = TRUE;
        break;
    default:
        assert(0);
    }

    if (parse_optional_double(line + 15, 12, &d1, 0.0) ||
        parse_optional_double(line + 28, 12, &d2, 0.0) ||
        parse_optional_double(line + 41, 7, &d3, 0.0) ||
        parse_optional_double(line + 49, 7, &d4, 0.0) ||
        parse_optional_uint  (line + 57, 3, &u1, 0)   ||
        parse_optional_uint  (line + 61, 3, &u2, 0)   ||
        parse_optional_double(line + 65, 4, &d5, 0.0) ||
        parse_optional_double(line + 70, 4, &d6, 0.0) ||
        parse_optional_double(line + 75, 7, &d7, 0.0) ||
        parse_optional_double(line + 83, 7, &d8, 0.0) ||
        parse_optional_uint  (line + 91, 2, &u3, 0)   ||
        parse_optional_double(line + 94, 3, &d9, 0.0) ||
        parse_optional_double(line + 98, 3, &d10, 0.0) ||
        parse_optional_double(line + 102, 3, &d11, 0.0) ||
        parse_optional_double(line + 106, 3, &d12, 0.0) ||
        parse_optional_double(line + 110, 6, &d13, 0.0) ||
        parse_optional_double(line + 117, 5, &d14, 0.0) ||
        parse_optional_double(line + 123, 6, &d15, 0.0) ||
        parse_optional_double(line + 130, 5, &d16, 0.0)) {
        return -1;
    }

    assert(d3 == 0.0 || (d3 >= -4418.0 && d3 <= 6544.2));
    assert(d4 == 0.0 || (d4 >= -5774.3 && d4 <= 10277.3));
    assert(u1 == 0 || (u1 >= 3 && u1 <= 183));
    assert(u2 == 0 || (u2 >= 1 && u2 <= 184));
    assert(d5 == 0.0 || (d5 >= 0.2 && d5 <= 11.5));
    assert(d6 == 0.0 || (d6 >= 0.2 && d6 <= 10.3));
    assert(d7 == 0.0 || (d7 >= 1915.95 && d7 <= 1992.53));
    assert(d8 == 0.0 || (d8 >= 1911.94 && d8 <= 1992.01));
    assert(u3 == 0 || (u3 >= 2 && u3 <= 36));
    assert(d9 == 0.0 || (d9 >= 0.0 && d9 <= 9.9));
    assert(d10 == 0.0 || (d10 >= 0.0 && d10 <= 9.9));
    assert(d11 == 0.0 || (d11 >= 0.0 && d11 <= 9.9));
    assert(d12 == 0.0 || (d12 >= 0.0 && d12 <= 9.9));
    assert(d13 == 0.0 || (d13 >= 2.183 && d13 <= 16.581));
    assert(d14 == 0.0 || (d14 >= 0.014 && d14 <= 1.977));
    assert(d15 == 0.0 || (d15 >= 1.905 && d15 <= 15.193));
    assert(d16 == 0.0 || (d16 >= 0.009 && d16 <= 1.468));

    // note: [units in Tycho-2 data file : units in struct]
    // if only one unit is given, it's the same in both.

    entry->mean_ra  = d1; // [deg]
    entry->mean_dec = d2; // [deg]
    entry->pm_ra  = d3 / 1000.0; // [mas/yr : arcsec/yr]
    entry->pm_dec = d4 / 1000.0; // [mas/yr : arcsec/yr]
    entry->sigma_mean_ra  = arcsec2deg((float)u1 / 1000.0); // [mas : deg]
    entry->sigma_mean_dec = arcsec2deg((float)u2 / 1000.0); // [mas : deg]
    entry->sigma_pm_ra  = d5 / 1000.0; // [mas/yr : arcsec/yr]
    entry->sigma_pm_dec = d6 / 1000.0; // [mas/yr : arcsec/yr]
    entry->epoch_mean_ra  = d7; // [yr]
    entry->epoch_mean_dec = d8; // [yr]
    entry->nobs = u3;
    entry->goodness_mean_ra = d9;
    entry->goodness_mean_dec = d10;
    entry->goodness_pm_ra = d11;
    entry->goodness_pm_dec = d12;
    entry->mag_BT = d13;   // [mag]
    entry->sigma_BT = d14; // [mag]
    entry->mag_VT = d15;   // [mag]
    entry->sigma_VT = d16; // [mag]

    if (parse_uint(line + 136, 3, &u1) ||
        parse_optional_uint(line + 142, 6, &u2, 0) ||
        parse_double(line + 152, 12, &d1) ||
        parse_double(line + 165, 12, &d2) ||
        parse_double(line + 178, 4, &d3) ||
        parse_double(line + 183, 4, &d4) ||
        parse_double(line + 188, 5, &d5) ||
        parse_double(line + 194, 5, &d6) ||
        parse_double(line + 202, 4, &d7)) {
        return -1;
    }

    memset(entry->hip_ccdm, 0, sizeof(entry->hip_ccdm));
    for (i=0; i<3; i++) {
        char c = line[148 + i];
        if (c == ' ')
            break;
        entry->hip_ccdm[i] = c;
    }
    tycho1 = line[140];
    posflag = line[200];

    assert(u1 >= 3 && u1 <= 999);
    assert(u2 == 0 || (u2 >= 1 && u2 <= 120404));
    assert(tycho1 == ' ' || tycho1 == 'T');
    assert(posflag == ' ' || posflag == 'D' || posflag == 'P');
    assert(d3 >= 0.81 && d3 <= 2.13);
    assert(d4 >= 0.72 && d4 <= 2.36);

    if (u1 == 999)
        entry->prox = -1.0;
    else
        entry->prox = arcsec2deg(u1 * 0.1); // [arcsec : deg]
    entry->tycho1_star = (tycho1 == 'T') ? TRUE : FALSE;
    entry->double_star = (posflag == 'D') ? TRUE : FALSE;
    entry->photo_center_treatment = (posflag == 'P') ? TRUE : FALSE;
    entry->hipparcos_id = u2;

    entry->ra  = d1; // [deg]
    entry->dec = d2; // [deg]
    entry->epoch_ra  = 1990.0 + d3; // [yr]
    entry->epoch_dec = 1990.0 + d4; // [yr]
    entry->sigma_ra  = arcsec2deg(d5 / 1000.0); // [mas : deg]
    entry->sigma_dec = arcsec2deg(d6 / 1000.0); // [mas : deg]
    entry->correlation = d7;

    return 0;
}

int tycho2_supplement_parse_entry(const char* line, tycho2_entry* entry) {
    int len;
    unsigned int t1, t2, t3;
    char htflag, tycho1, bvhflag;
    double d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12;
    unsigned int u1, u2;

    memset(entry, 0, sizeof(tycho2_entry));

    for (len=0; len<TYCHO_SUPPLEMENT_RECORD_SIZE_RAW; len++)
        if (!line[len]) break;
    if (len != TYCHO_SUPPLEMENT_RECORD_SIZE_RAW) {
        fprintf(stderr, "Tycho2: couldn't parse supplement record: length %i, not %i.\n",
                len, TYCHO_SUPPLEMENT_RECORD_SIZE_RAW);
        return -1;
    }

    if (parse_uint(line + 0, 4, &t1) ||
        parse_uint(line + 5, 5, &t2) ||
        parse_uint(line + 11, 1, &t3)) {
        return -1;
    }
    assert(t1 >= 2);
    assert(t1 <= 9529);
    assert(t2 >= 1);
    assert(t2 <= 12112);
    assert(t3 >= 1);
    assert(t3 <= 4);

    entry->tyc1 = t1;
    entry->tyc2 = t2;
    entry->tyc3 = t3;

    htflag = line[13];
    switch (htflag) {
    case 'H':
        entry->hipparcos_star = TRUE;
        break;
    case 'T':
        entry->tycho1_star = TRUE;
        break;
    default:
        assert(0);
    }

    bvhflag = line[81];
    switch (bvhflag) {
    case 'B':
    case 'V':
    case 'H':
    case ' ':
        break;
    default:
        assert(0);
    }

    tycho1 = line[113];
    switch (tycho1) {
    case ' ':
    case 'T':
        break;
    default:
        assert(0);
    }

    if (parse_double(line + 15, 12, &d1) ||      // RAdeg
        parse_double(line + 28, 12, &d2) ||      // DEdeg
        parse_optional_double(line + 41, 7, &d3, 0.0) ||  // pmRA
        parse_optional_double(line + 49, 7, &d4, 0.0) ||  // pmDE
        parse_double(line + 57, 5, &d5) ||       // e_RA*
        parse_double(line + 63, 5, &d6) ||       // e_DE
        parse_optional_double(line + 69, 5, &d7, 0.0) ||  // e_pmRA
        parse_optional_double(line + 75, 5, &d8, 0.0) ||  // e_pmDE
        parse_optional_double(line + 83, 6, &d9, 0.0) ||  // BT
        parse_optional_double(line + 90, 5, &d10, 0.0) || // e_BT
        parse_optional_double(line + 96, 6, &d11, 0.0) || // VT
        parse_optional_double(line + 103, 5, &d12, 0.0) || // e_VT
        parse_optional_uint  (line + 109, 3, &u1, 0)   || // prox
        parse_optional_uint  (line + 115, 6, &u2, 0)) {
        return -1;
    }
    grab_substring(entry->hip_ccdm, line + 121, 1);

    assert(u2 == 0 || (u2 >= 1 && u2 <= 120404));

    entry->ra  = d1; // [deg]
    entry->dec = d2; // [deg]
    entry->pm_ra  = d3 / 1000.0; // [mas/yr : arcsec/yr]
    entry->pm_dec = d4 / 1000.0; // [mas/yr : arcsec/yr]
    entry->sigma_ra  = arcsec2deg(d5 / 1000.0); // [arcsec : deg]
    entry->sigma_dec = arcsec2deg(d6 / 1000.0); // [arcsec : deg]
    entry->sigma_pm_ra  = d7 / 1000.0; // [mas/yr : arcsec/yr]
    entry->sigma_pm_dec = d8 / 1000.0; // [mas/yr : arcsec/yr]
    if (bvhflag == 'H') {
        entry->mag_HP = d11;   // [mag]
        entry->sigma_HP = d12; // [mag]
    } else {
        entry->mag_BT = d9;    // [mag]
        entry->sigma_BT = d10; // [mag]
        entry->mag_VT = d11;   // [mag]
        entry->sigma_VT = d12; // [mag]
    }
    entry->prox = arcsec2deg(u1 * 0.1); // [0.1 arcsec : deg]
    entry->hipparcos_id = u2;

    return 0;
}

