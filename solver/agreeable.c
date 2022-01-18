/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "os-features.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "matchobj.h"
#include "matchfile.h"
#include "solvedclient.h"
#include "solvedfile.h"
#include "boilerplate.h"

char* OPTIONS = "hA:B:I:J:L:M:r:f:s:S:Fa";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "Usage: %s [options]\n"
            "   [-A first-field]\n"
            "   [-B last-field]\n"
            "   [-I first-field-filenum]\n"
            "   [-J last-field-filenum]\n"
            "   [-L write-leftover-matches-file]\n"
            "   [-M write-successful-matches-file]\n"
            "   [-r ratio-needed-to-solve]\n"
            "   [-f minimum-field-objects-needed-to-solve] (default: no minimum)\n"
            "   (      [-F]: write out the first sufficient match to surpass the solve threshold.\n"
            "     or   [-a]: write out all matches passing the solve threshold.\n"
            "          (default is to write out the single best match (largest ratio))\n"
            "   )\n"
            "   [-S <solved-file-template>]\n"
            "   <input-match-file> ...\n"
            "\n", progname);
}

static void write_field(pl* agreeing,
                        pl* leftover,
                        int fieldfile,
                        int fieldnum);

char* leftoverfname = NULL;
matchfile* leftovermf = NULL;
char* agreefname = NULL;
matchfile* agreemf = NULL;
il* solved;
il* unsolved;
char* solvedfile = NULL;
double ratio_tosolve = 0.0;
int ninfield_tosolve = 0;


enum modes {
    MODE_BEST,
    MODE_FIRST,
    MODE_ALL
};

int main(int argc, char *argv[]) {
    int argchar;
    char* progname = argv[0];
    char** inputfiles = NULL;
    int ninputfiles = 0;
    int i;
    int firstfield=0, lastfield=INT_MAX-1;
    int firstfieldfile=1, lastfieldfile=INT_MAX-1;
    matchfile** mfs;
    MatchObj** mos;
    anbool* eofs;
    anbool* eofieldfile;
    int nread = 0;
    int f;
    int fieldfile;
    int totalsolved, totalunsolved;
    int mode = MODE_BEST;
    double logodds_tosolve = -LARGE_VAL;
    anbool agree = FALSE;

    MatchObj* bestmo;
    bl* keepers;
    bl* leftovers = NULL;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1) {
        switch (argchar) {
        case 'S':
            solvedfile = optarg;
            break;
        case 'F':
            mode = MODE_FIRST;
            break;
        case 'a':
            mode = MODE_ALL;
            break;
        case 'r':
            ratio_tosolve = atof(optarg);
            logodds_tosolve = log(ratio_tosolve);
            break;
        case 'f':
            ninfield_tosolve = atoi(optarg);
            break;
        case 'M':
            agreefname = optarg;
            break;
        case 'L':
            leftoverfname = optarg;
            break;
        case 'I':
            firstfieldfile = atoi(optarg);
            break;
        case 'J':
            lastfieldfile = atoi(optarg);
            break;
        case 'A':
            firstfield = atoi(optarg);
            break;
        case 'B':
            lastfield = atoi(optarg);
            break;
        case 'h':
        default:
            printHelp(progname);
            exit(-1);
        }
    }
    if (optind < argc) {
        ninputfiles = argc - optind;
        inputfiles = argv + optind;
    } else {
        printHelp(progname);
        exit(-1);
    }

    if (lastfield < firstfield) {
        fprintf(stderr, "Last field (-B) must be at least as big as first field (-A)\n");
        exit(-1);
    }

    if (leftoverfname) {
        leftovermf = matchfile_open_for_writing(leftoverfname);
        if (!leftovermf) {
            fprintf(stderr, "Failed to open file %s to write leftover matches.\n", leftoverfname);
            exit(-1);
        }
        BOILERPLATE_ADD_FITS_HEADERS(leftovermf->header);
        qfits_header_add(leftovermf->header, "HISTORY", "This file was created by the program \"agreeable\".", NULL, NULL);
        if (matchfile_write_headers(leftovermf)) {
            fprintf(stderr, "Failed to write leftovers matchfile header.\n");
            exit(-1);
        }
        leftovers = bl_new(256, sizeof(MatchObj));
    }
    if (agreefname) {
        agreemf = matchfile_open_for_writing(agreefname);
        if (!agreemf) {
            fprintf(stderr, "Failed to open file %s to write agreeing matches.\n", agreefname);
            exit(-1);
        }
        BOILERPLATE_ADD_FITS_HEADERS(agreemf->header);
        qfits_header_add(agreemf->header, "HISTORY", "This file was created by the program \"agreeable\".", NULL, NULL);
        if (matchfile_write_headers(agreemf)) {
            fprintf(stderr, "Failed to write agreeing matchfile header.\n");
            exit(-1);
        }
        agree = TRUE;
    }

    solved = il_new(256);
    unsolved = il_new(256);

    keepers = bl_new(256, sizeof(MatchObj));

    totalsolved = totalunsolved = 0;

    mos =  calloc(ninputfiles, sizeof(MatchObj*));
    eofs = calloc(ninputfiles, sizeof(anbool));
    eofieldfile = malloc(ninputfiles * sizeof(anbool));
    mfs = malloc(ninputfiles * sizeof(matchfile*));

    for (i=0; i<ninputfiles; i++) {
        fprintf(stderr, "Opening file %s...\n", inputfiles[i]);
        mfs[i] = matchfile_open(inputfiles[i]);
        if (!mfs[i]) {
            fprintf(stderr, "Failed to open matchfile %s.\n", inputfiles[i]);
            exit(-1);
        }
    }

    // we assume the matchfiles are sorted by field id and number.
    for (fieldfile=firstfieldfile; fieldfile<=lastfieldfile; fieldfile++) {
        anbool alldone = TRUE;

        memset(eofieldfile, 0, ninputfiles * sizeof(anbool));

        for (f=firstfield; f<=lastfield; f++) {
            int fieldnum = f;
            anbool donefieldfile;
            anbool solved_it;
            bl* writematches = NULL;

            // quit if we've reached the end of all the input files.
            alldone = TRUE;
            for (i=0; i<ninputfiles; i++)
                if (!eofs[i]) {
                    alldone = FALSE;
                    break;
                }
            if (alldone)
                break;

            // move on to the next fieldfile if all the input files have been
            // exhausted.
            donefieldfile = TRUE;
            for (i=0; i<ninputfiles; i++)
                if (!eofieldfile[i] && !eofs[i]) {
                    donefieldfile = FALSE;
                    break;
                }
            if (donefieldfile)
                break;

            // start a new field.
            fprintf(stderr, "File %i, Field %i.\n", fieldfile, f);
            solved_it = FALSE;
            bestmo = NULL;

            for (i=0; i<ninputfiles; i++) {
                int nr = 0;
                int ns = 0;

                while (1) {
                    if (eofs[i])
                        break;
                    if (!mos[i])
                        mos[i] = matchfile_read_match(mfs[i]);
                    if (unlikely(!mos[i])) {
                        eofs[i] = TRUE;
                        break;
                    }

                    // skip past entries that are out of range...
                    if ((mos[i]->fieldfile < firstfieldfile) ||
                        (mos[i]->fieldfile > lastfieldfile) ||
                        (mos[i]->fieldnum < firstfield) ||
                        (mos[i]->fieldnum > lastfield)) {
                        mos[i] = NULL;
                        ns++;
                        continue;
                    }

                    if (mos[i]->fieldfile > fieldfile)
                        eofieldfile[i] = TRUE;

                    if (mos[i]->fieldfile != fieldfile)
                        break;

                    assert(mos[i]->fieldnum >= fieldnum);
                    if (mos[i]->fieldnum != fieldnum)
                        break;
                    nread++;
                    if (nread % 10000 == 9999) {
                        fprintf(stderr, ".");
                        fflush(stderr);
                    }

                    // if we've already found a solution, skip past the
                    // remaining matches in this file...
                    if (solved_it && (mode == MODE_FIRST)) {
                        ns++;
                        mos[i] = NULL;
                        continue;
                    }

                    nr++;

                    if ((mos[i]->logodds >= logodds_tosolve)  &&
                        (mos[i]->nindex >= ninfield_tosolve)) {
                        solved_it = TRUE;
                        // (note, we get a pointer to its position in the list)
                        mos[i] = bl_append(keepers, mos[i]);
                        if (!bestmo || mos[i]->logodds > bestmo->logodds)
                            bestmo = mos[i];
                    } else if (leftovers) {
                        bl_append(leftovers, mos[i]);
                    }

                    mos[i] = NULL;

                }
                if (nr || ns)
                    fprintf(stderr, "File %s: read %i matches, skipped %i matches.\n", inputfiles[i], nr, ns);
            }

            // which matches do we want to write out?
            if (agree) {
                writematches = bl_new(256, sizeof(MatchObj));

                switch (mode) {
                case MODE_BEST:
                case MODE_FIRST:
                    if (bestmo)
                        bl_append(writematches, bestmo);
                    break;
                case MODE_ALL:
                    bl_append_list(writematches, keepers);
                    break;
                }
            }

            write_field(writematches, leftovers, fieldfile, fieldnum);

            if (agree)
                bl_free(writematches);

            if (leftovers)
                bl_remove_all(leftovers);
            if (keepers)
                bl_remove_all(keepers);

            fprintf(stderr, "This file: %i fields solved, %i unsolved.\n", il_size(solved), il_size(unsolved));
            fprintf(stderr, "Grand total: %i solved, %i unsolved.\n", totalsolved + il_size(solved), totalunsolved + il_size(unsolved));
        }
        totalsolved += il_size(solved);
        totalunsolved += il_size(unsolved);
		
        il_remove_all(solved);
        il_remove_all(unsolved);

        if (alldone)
            break;
    }

    for (i=0; i<ninputfiles; i++)
        matchfile_close(mfs[i]);
    free(mfs);
    free(mos);
    free(eofs);

    fprintf(stderr, "\nRead %i matches.\n", nread);
    fflush(stderr);

    if (keepers)
        bl_free(keepers);
    if (leftovers)
        bl_free(leftovers);

    il_free(solved);
    il_free(unsolved);

    if (leftovermf) {
        matchfile_fix_headers(leftovermf);
        matchfile_close(leftovermf);
    }

    if (agreemf) {
        matchfile_fix_headers(agreemf);
        matchfile_close(agreemf);
    }

    return 0;
}

static void write_field(bl* agreeing,
                        bl* leftover,
                        int fieldfile,
                        int fieldnum) {
    int i;

    if (!bl_size(agreeing))
        il_append(unsolved, fieldnum);
    else {
        il_append(solved, fieldnum);
        if (solvedfile) {
            char fn[256];
            sprintf(fn, solvedfile, fieldfile);
            solvedfile_set(fn, fieldnum);
        }
    }

    for (i=0; agreeing && i<bl_size(agreeing); i++) {
        MatchObj* mo = bl_access(agreeing, i);
        if (matchfile_write_match(agreemf, mo))
            fprintf(stderr, "Error writing an agreeing match.");
        fprintf(stderr, "Field %i: Logodds %g (%g)\n", fieldnum, mo->logodds, exp(mo->logodds));
    }

    if (leftover && bl_size(leftover)) {
        fprintf(stderr, "Field %i: writing %i leftovers...\n", fieldnum,
                bl_size(leftover));
        for (i=0; i<bl_size(leftover); i++) {
            MatchObj* mo = bl_access(leftover, i);
            if (matchfile_write_match(leftovermf, mo))
                fprintf(stderr, "Error writing a leftover match.");
        }
    }
}
