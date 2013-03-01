/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <regex.h>

#include "qfits.h"
#include "qfits_error.h"
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
	// SIP
	"^[AB]P?_ORDER$",
	"^[AB]P?_[[:digit:]]_[[:digit:]]$",
	// Other
	"^PV[[:digit:]]*_[[:digit:]]*.?$",
	"^CROTA[12]$",
	"^END$",
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

int new_wcs(const char* infn, const char* wcsfn, const char* outfn,
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

	inhdr = qfits_header_read(infn);
	if (!inhdr) {
		ERROR("Failed to read FITS header from input file \"%s\"", infn);
        goto bailout;
	}
	wcshdr = qfits_header_read(wcsfn);
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

	N = qfits_header_n(inhdr);
	for (i=0; i<N; i++) {
        char line[FITS_LINESZ + 1];
		if (qfits_header_getitem(inhdr, i, key, val, comment, line)) {
			ERROR("Failed to read FITS header card %i from input file", i);
            goto bailout;
		}

		if (key_matches(key, re1, exclude_input, NE1, &imatch)) {
			logverb("Regular expression matched: \"%s\", key \"%s\".\n", exclude_input[imatch], key);
			snprintf(newkey, FITS_LINESZ+1, "Original key: \"%s\"", key);
			qfits_header_append(outhdr, "COMMENT", newkey, NULL, NULL);
            // Completely skip the END card, since _ND is not a valid line.
            if (streq(key, "END"))
                continue;
            line[0] = '_';
		}

		qfits_header_append(outhdr, key, val, comment, line);
	}
	qfits_header_destroy(inhdr);
    inhdr = NULL;

	logverb("Reading WCS file FITS headers...\n");

	qfits_header_append(outhdr, "COMMENT", "", NULL, NULL);
	qfits_header_append(outhdr, "COMMENT", "--Start of Astrometry.net WCS solution--", NULL, NULL);
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
		qfits_header_append(outhdr, key, val, comment, line);
	}
	qfits_header_destroy(wcshdr);
    wcshdr = NULL;

	qfits_header_append(outhdr, "COMMENT", "", NULL, NULL);
	qfits_header_append(outhdr, "COMMENT", "--End of Astrometry.net WCS--", NULL, NULL);
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
		if (qfits_get_datinfo(infn, 0, &datstart, &datsize)) {
			ERROR("Couldn't find size of FITS data block.");
            goto bailout;
		}
        infid = fopen(infn, "rb");
        if (!infid) {
            SYSERROR("Failed to open input file \"%s\"", infn);
            goto bailout;
        }
        logverb("Copying from offset %i to offset %i (length %i) of the input file to the output.\n",
                datstart, datstart + datsize, datsize);
        if (pipe_file_offset(infid, datstart, datsize, outfid)) {
            ERROR("Failed to copy the data block");
            fclose(infid);
            goto bailout;
        }
        fclose(infid);
		if (fits_pad_file(outfid)) {
            ERROR("Failed to pad FITS file \"%s\"", outfn);
            goto bailout;
        }
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
