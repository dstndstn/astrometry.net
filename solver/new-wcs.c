/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <regex.h>

#include "anqfits.h"
#include "an-bool.h"
#include "fitsioutils.h"
#include "ioutils.h"
#include "new-wcs.h"
#include "errors.h"
#include "log.h"

static char* exclude_input[] = {
    // TAN
    "^CTYPE[12]$",
    "^WCSAXES$",
    "^EQUINOX$",
    "^LONPOLE$",
    "^LATPOLE$",
    "^CRVAL[12]$",
    "^CRPIX[12]$",
    "^CUNIT[12]$",
    "^CD[12]_[12]$",
    "^CDELT[12]$",
    "^PC[12]_[12]$",
    // SIP
    "^[AB]P?_ORDER$",
    "^[AB]P?_[[:digit:]]_[[:digit:]]$",
    // Other
    "^PV[[:digit:]]*_[[:digit:]]*.?$",
    "^CROTA[12]$",
    "^END$",
    // our TAN/SIP
    "^IMAGE[HW]$",
    // Pinpoint proprietary distortion
    "^TR[12]_[[:digit:]]+$",
};
static int NE1 = sizeof(exclude_input) / sizeof(char*);

static char* exclude_wcs[] = {
    "^SIMPLE$",
    "^BITPIX$",
    "^EXTEND$",
    "^NAXIS$",
    "^END$",
};
static int NE2 = sizeof(exclude_wcs) / sizeof(char*);

static anbool key_matches(char* key, regex_t* res, char** re_strings, int NE, int* rematched) {
    int e;
    for (e=0; e<NE; e++) {
        regmatch_t match[1];
        int errcode;
        errcode = regexec(res + e, key, sizeof(match)/sizeof(regmatch_t), match, 0);
        if (errcode == REG_NOMATCH)
            continue;
        if (errcode) {
            char err[256];
            regerror(errcode, res + e, err, sizeof(err));
            fprintf(stderr, "Failed to match regular expression \"%s\" with string \"%s\": %s\n", re_strings[e], key, err);
            exit(-1);
        }
        if (rematched)
            *rematched = e;
        return TRUE;
    }
    return FALSE;
}

int new_wcs(const char* infn, int extension,
            const char* wcsfn, const char* outfn,
            anbool copydata) {
    FILE* outfid = NULL;
    int i, N;
    int e;
    regex_t re1[NE1];
    regex_t re2[NE2];
    qfits_header *inhdr=NULL, *outhdr=NULL, *wcshdr=NULL;

    char key[FITS_LINESZ + 1];
    char newkey[FITS_LINESZ + 1];
    char val[FITS_LINESZ + 1];
    char comment[FITS_LINESZ + 1];
    int imatch = -1;
    // how many REs have successfully been compiled.
    int n1=0, n2=0;

    outfid = fopen(outfn, "wb");
    if (!outfid) {
        SYSERROR("Failed to open output file \"%s\"", outfn);
        goto bailout;
    }

    inhdr = anqfits_get_header2(infn, extension);
    if (!inhdr) {
        ERROR("Failed to read FITS header from input file \"%s\" ext %i",
              infn, extension);
        goto bailout;
    }
    wcshdr = anqfits_get_header2(wcsfn, 0);
    if (!wcshdr) {
        ERROR("Failed to read FITS header from WCS file \"%s\"", wcsfn);
        goto bailout;
    }

    outhdr = qfits_header_new();
    if (!outhdr) {
        ERROR("Failed to allocate new output FITS header.");
        goto bailout;
    }

    // Compile regular expressions...
    for (e=0; e<NE1; e++) {
        int errcode;
        errcode = regcomp(re1 + e, exclude_input[e], REG_EXTENDED);
        if (errcode) {
            char err[256];
            regerror(errcode, re1 + e, err, sizeof(err));
            ERROR("Failed to compile regular expression \"%s\": %s", exclude_input[e], err);
            goto bailout;
        }
        n1++;
    }
    for (e=0; e<NE2; e++) {
        int errcode;
        errcode = regcomp(re2 + e, exclude_wcs[e], REG_EXTENDED);
        if (errcode) {
            char err[256];
            regerror(errcode, re2 + e, err, sizeof(err));
            ERROR("Failed to compile regular expression \"%s\": %s", exclude_wcs[e], err);
            goto bailout;
        }
        n2++;
    }

    logverb("Reading input file FITS headers...\n");

    if (extension) {
        // Copy the primary header unchanged
        qfits_header* phdr = anqfits_get_header2(infn, 0);
        if (!phdr) {
            ERROR("Failed to read primary FITS header from input file \"%s\n",
                  infn);
            goto bailout;
        }
        if (qfits_header_dump(phdr, outfid) ||
            fits_pad_file(outfid)) {
            SYSERROR("Failed to write primary header to file %s", outfn);
            goto bailout;
        }
        qfits_header_destroy(phdr);
    }
    
    N = qfits_header_n(inhdr);
    for (i=0; i<N; i++) {
        anbool added_newkey = FALSE;
        char line[FITS_LINESZ + 1];
        if (qfits_header_getitem(inhdr, i, key, val, comment, line)) {
            ERROR("Failed to read FITS header card %i from input file", i);
            goto bailout;
        }
        logverb("Read input header line: \"%s\"\n", line);

        if (key_matches(key, re1, exclude_input, NE1, &imatch)) {
            logverb("Regular expression matched: \"%s\", key \"%s\".\n", exclude_input[imatch], key);
            snprintf(newkey, FITS_LINESZ, "Original key: \"%s\"", key);
            qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
            // Completely skip the END card, since _ND is not a valid line.
            if (streq(key, "END"))
                continue;
            snprintf(newkey, FITS_LINESZ, "_%.7s", key+1);
            logverb("New key: \"%s\"\n", newkey);
            strcpy(key, newkey);
            line[0] = '_';
            added_newkey = TRUE;
        }
        // If the header already contains this new (starting-with-"_")
        // key, add three comment cards instead.
        if (starts_with(key, "_") &&
            (qfits_header_getstr(inhdr, key) ||
             qfits_header_getstr(outhdr, key))) {
            logverb("Key \"%s\" already exists; adding COMMENT cards for value and comment instead\n", key);
            if (!added_newkey) {
                snprintf(newkey, FITS_LINESZ, "Original key: \"%s\"", key);
                qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
            }
            snprintf(newkey, FITS_LINESZ, " = %s", val);
            qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
            snprintf(newkey, FITS_LINESZ, " / %s", comment);
            qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
            continue;
        }

        qfits_header_append(outhdr, key, val, comment, line);
    }
    qfits_header_destroy(inhdr);
    inhdr = NULL;

    logverb("Reading WCS file FITS headers...\n");

    qfits_header_append(outhdr, "COMMENT", "", NULL, NULL);
    qfits_header_append(outhdr, "COMMENT", "--Start of Astrometry.net WCS solution--", NULL, NULL);
    qfits_header_append(outhdr, "COMMENT", "--Put in by the new-wcs program--", NULL, NULL);
    qfits_header_append(outhdr, "COMMENT", "", NULL, NULL);

    N = qfits_header_n(wcshdr);
    for (i=0; i<N; i++) {
        char line[FITS_LINESZ + 1];
        if (qfits_header_getitem(wcshdr, i, key, val, comment, line)) {
            ERROR("Failed to read FITS header card %i from WCS file.", i);
            goto bailout;
        }

        if (key_matches(key, re2, exclude_wcs, NE2, &imatch)) {
            logverb("Regular expression matched: \"%s\", key \"%s\".\n", exclude_wcs[imatch], key);
            // These don't really need to appear in the output file...
            /*
             snprintf(newkey, FITS_LINESZ+1, "Original WCS key: \"%s\"", key);
             qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
             snprintf(newkey, FITS_LINESZ+1, "_%.7s", key);
             strcpy(key, newkey);
             */
            continue;
        }
        if (streq(key, "DATE") && qfits_header_getstr(outhdr, key)) {
            // If the input header already had a DATE card,
            snprintf(newkey, FITS_LINESZ, "Original WCS key: \"%s\"", key);
            qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
            snprintf(newkey, FITS_LINESZ, "_%.7s", key);
            strcpy(key, newkey);
            line[0] = '_';
        }

        qfits_header_append(outhdr, key, val, comment, line);
    }
    qfits_header_destroy(wcshdr);
    wcshdr = NULL;

    qfits_header_append(outhdr, "COMMENT", "", NULL, NULL);
    qfits_header_append(outhdr, "COMMENT", "--End of Astrometry.net WCS--", NULL, NULL);
    qfits_header_append(outhdr, "COMMENT", "--(Put in by the new-wcs program)--", NULL, NULL);
    qfits_header_append(outhdr, "COMMENT", "", NULL, NULL);


    qfits_header_append(outhdr, "END", NULL, NULL, NULL);

    if (qfits_header_dump(outhdr, outfid) ||
        fits_pad_file(outfid)) {
        SYSERROR("Failed to write output header to file %s", outfn);
        goto bailout;
    }
    qfits_header_destroy(outhdr);
    outhdr = NULL;

    if (copydata) {
        int datsize, datstart;
        FILE* infid = NULL;
        anqfits_t* anq = NULL;
        anq = anqfits_open(infn);
        if (!anq) {
            ERROR("Failed to open file \"%s\"", infn);
            goto bailout;
        }
        datstart = anqfits_data_start(anq, extension);
        datsize  = anqfits_data_size (anq, extension);
        infid = fopen(infn, "rb");
        if (!infid) {
            SYSERROR("Failed to open input file \"%s\"", infn);
            anqfits_close(anq);
            goto bailout;
        }
        logverb("Copying from offset %i to offset %i (length %i) of the input file to the output.\n",
                datstart, datstart + datsize, datsize);
        if (pipe_file_offset(infid, datstart, datsize, outfid)) {
            ERROR("Failed to copy the data block");
            fclose(infid);
            anqfits_close(anq);
            goto bailout;
        }
        fclose(infid);
        if (fits_pad_file(outfid)) {
            ERROR("Failed to pad FITS file \"%s\"", outfn);
            anqfits_close(anq);
            goto bailout;
        }
        anqfits_close(anq);
        anq = NULL;
    }
    
    if (fclose(outfid)) {
        SYSERROR("Failed to close output file \"%s\"", outfn);
        goto bailout;
    }


    // Free regular expressions...
    for (e=0; e<NE1; e++)
        regfree(re1 + e);
    for (e=0; e<NE2; e++)
        regfree(re2 + e);
    return 0;

 bailout:
    if (outfid)
        fclose(outfid);
    if (inhdr)
        qfits_header_destroy(inhdr);
    if (outhdr)
        qfits_header_destroy(outhdr);
    if (wcshdr)
        qfits_header_destroy(wcshdr);
    for (i=0; i<n1; i++)
        regfree(re1 + i);
    for (i=0; i<n2; i++)
        regfree(re2 + i);
    return -1;
}
