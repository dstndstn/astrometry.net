/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "os-features.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "matchobj.h"
#include "matchfile.h"
#include "rdlist.h"
#include "solvedfile.h"

char* OPTIONS = "hR:A:B:n:t:f:C:T:F:i:I:m:M:";

void printHelp(char* progname) {
    fprintf(stderr, "Usage: %s\n"
            "   -R rdls-file-template\n"
            "   [-i <first-file>]\n"
            "   [-I <last-file>]\n"
            "   [-A <first-field>]  (default 1)\n"
            "   [-B <last-field>]   (default the largest field encountered)\n"
            "   [-n <negative-fields-rdls>]"
            "   [-f <false-positive-fields-rdls>]\n"
            "   [-t <true-positive-fields-rdls>]\n"
            "   [-C <number-of-RDLS-stars-to-compute-center>]\n"
            "   [-T <true-positive-solvedfile>]\n"
            "   [-F <false-positive-solvedfile>]  (note, both of these are per-MATCH, not per-FIELD)\n"
            "   [-m <true matches output file>]\n"
            "   [-M <false matches output file>]\n"
            "\n"
            "   <input-match-file> ...\n"
            "\n", progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* progname = argv[0];
    char** inputfiles = NULL;
    int ninputfiles = 0;
    char* rdlsfname = NULL;
    rdlist_t* rdls = NULL;
    int i;
    int correct, incorrect;
    int firstfield = 1;
    int lastfield = -1;

    int nfields;

    int Ncenter = 0;

    char* fpsolved = NULL;
    char* tpsolved = NULL;
    int nfields_total;

    int firstfileid = 0;
    int lastfileid = 0;
    int fileid;

    matchfile** matchfiles;
    int* mfcursors;

    char* truematchfn = NULL;
    char* falsematchfn = NULL;
    matchfile* truematch = NULL;
    matchfile* falsematch = NULL;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1) {
        switch (argchar) {
        case 'h':
            printHelp(progname);
            return (HELP_ERR);
        case 'm':
            truematchfn = optarg;
            break;
        case 'M':
            falsematchfn = optarg;
            break;
        case 'A':
            firstfield = atoi(optarg);
            break;
        case 'B':
            lastfield = atoi(optarg);
            break;
        case 'C':
            Ncenter = atoi(optarg);
            break;
        case 'R':
            rdlsfname = optarg;
            break;
        case 'T':
            tpsolved = optarg;
            break;
        case 'F':
            fpsolved = optarg;
            break;
        case 'i':
            firstfileid = atoi(optarg);
            break;
        case 'I':
            lastfileid = atoi(optarg);
            break;
        default:
            return (OPT_ERR);
        }
    }
    if (optind < argc) {
        ninputfiles = argc - optind;
        inputfiles = argv + optind;
    } else {
        printHelp(progname);
        exit(-1);
    }
    if (!rdlsfname) {
        fprintf(stderr, "You must specify an RDLS file!\n");
        printHelp(progname);
        exit(-1);
    }

    if (truematchfn) {
        truematch = matchfile_open_for_writing(truematchfn);
        if (!truematch) {
            fprintf(stderr, "Failed to open file %s for writing matches.\n", truematchfn);
            exit(-1);
        }
        if (matchfile_write_headers(truematch)) {
            fprintf(stderr, "Failed to write header for %s\n", truematchfn);
            exit(-1);
        }
    }
    if (falsematchfn) {
        falsematch = matchfile_open_for_writing(falsematchfn);
        if (!falsematch) {
            fprintf(stderr, "Failed to open file %s for writing matches.\n", falsematchfn);
            exit(-1);
        }
        if (matchfile_write_headers(falsematch)) {
            fprintf(stderr, "Failed to write header for %s\n", falsematchfn);
            exit(-1);
        }
    }

    matchfiles = malloc(ninputfiles * sizeof(matchfile*));
    mfcursors = calloc(ninputfiles, sizeof(int));

    for (i=0; i<ninputfiles; i++) {
        char* fname = inputfiles[i];
        printf("Opening matchfile %s...\n", fname);
        matchfiles[i] = matchfile_open(fname);
        if (!matchfiles[i]) {
            fprintf(stderr, "Failed to open matchfile %s.\n", fname);
            exit(-1);
        }
    }

    correct = incorrect = 0;
    nfields_total = 0;

    for (i=0; i<ninputfiles; i++) {
        matchfile* mf;
        MatchObj* mo;
        mf = matchfiles[i];

        for (fileid=firstfileid; fileid<=lastfileid; fileid++) {
            char fn[1024];
            int nread = 0;
            sprintf(fn, rdlsfname, fileid);
            //printf("Reading rdls file \"%s\"...\n", fn);
            fflush(stdout);
            rdls = rdlist_open(fn);
            if (!rdls) {
                fprintf(stderr, "Couldn't read rdls file.\n");
                exit(-1);
            }

            nfields = rdlist_n_fields(rdls);
            //printf("Read %i fields from rdls file.\n", nfields);
            if ((lastfield != -1) && (nfields > lastfield)) {
                nfields = lastfield + 1;
            } else {
                lastfield = nfields;
            }

            for (; mfcursors[i]<matchfile_count(mf); mfcursors[i]++) {
                int filenum;
                int fieldnum;
                double rac, decc;
                double r2;
                double arc;
                int nrd;
                rd_t* rd;
                int k;
                anbool err = FALSE;

                mo = matchfile_read_match(mf);
                filenum = mo->fieldfile;
                if (filenum < fileid)
                    continue;
                if (filenum > fileid) {
                    matchfile_pushback_match(mf);
                    break;
                }
                fieldnum = mo->fieldnum;
                if (fieldnum < firstfield)
                    continue;
                if (fieldnum > lastfield)
                    continue;

                nread++;

                rd = rdlist_read_field_num(rdls, fieldnum, NULL);
                if (!rd) {
                    fprintf(stderr, "Failed to read RDLS entries for field %i.\n", fieldnum);
                    exit(-1);
                }
                nrd = rd_n(rd);
                if (Ncenter)
                    nrd = MIN(nrd, Ncenter);

                r2 = square(mo->radius);
                arc = deg2arcmin(mo->radius_deg);
                xyzarr2radec(mo->center, &rac, &decc);

                for (k=0; k<nrd; k++) {
                    double xyz[3];
                    double ra, dec;
                    ra  = rd_getra (rd, k);
                    dec = rd_getdec(rd, k);
                    radecdeg2xyzarr(ra, dec, xyz);
                    if (distsq_exceeds(xyz, mo->center, 3, r2 * 1.2)) {
                        printf("\nError: Field %i: match says center (%g, %g), scale %g arcmin, but\n",
                               fieldnum, rac, decc, arc);
                        printf("rdls %i is (%g, %g).\n", k, ra, dec);
                        printf("Logprob %g (%g).\n", mo->logodds, exp(mo->logodds));
                        err = TRUE;
                        break;
                    }
                }
                rd_free(rd);

                if (err) {
                    incorrect++;
                    if (falsematch) {
                        if (matchfile_write_match(falsematch, mo)) {
                            fprintf(stderr, "Failed to write match to %s\n", falsematchfn);
                            exit(-1);
                        }
                    }
                } else {
                    printf("Field %5i: correct hit: (%8.3f, %8.3f), scale %6.3f arcmin, logodds %g (%g)\n",
                           fieldnum, rac, decc, arc, mo->logodds, exp(mo->logodds));
                    correct++;
                    if (truematch) {
                        if (matchfile_write_match(truematch, mo)) {
                            fprintf(stderr, "Failed to write match to %s\n", truematchfn);
                            exit(-1);
                        }
                    }
                }
                fflush(stdout);

                if (tpsolved && !err)
                    solvedfile_set(tpsolved, nfields_total);
                if (fpsolved && err)
                    solvedfile_set(fpsolved, nfields_total);

                nfields_total++;
            }

            rdlist_close(rdls);

            printf("Read %i from %s for fileid %i\n", nread, inputfiles[i], fileid);
        }
    }

    printf("\n");
    printf("Read a total of %i correct and %i incorrect matches.\n", correct, incorrect);

    for (i=0; i<ninputfiles; i++) {
        matchfile_close(matchfiles[i]);
    }
    free(matchfiles);
    free(mfcursors);

    if (tpsolved)
        solvedfile_setsize(tpsolved, nfields_total);
    if (fpsolved)
        solvedfile_setsize(fpsolved, nfields_total);

    if (truematch) {
        if (matchfile_fix_headers(truematch) ||
            matchfile_close(truematch)) {
            fprintf(stderr, "Failed to fix header for %s\n", truematchfn);
            exit(-1);
        }
    }
    if (falsematch) {
        if (matchfile_fix_headers(falsematch) ||
            matchfile_close(falsematch)) {
            fprintf(stderr, "Failed to fix header for %s\n", falsematchfn);
            exit(-1);
        }
    }

    return 0;
}

    
